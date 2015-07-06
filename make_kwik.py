"""Conversion script from template matching files to Kwik.

There are several improvements:

* Don't load all features and masks in memory (need for KwikCreator to accept
  features and masks as generators, not lists).

"""

import os
import os.path as op
import shutil

import numpy as np

from phy.cluster.algorithms import SpikeDetekt
from phy.electrode import load_probe
from phy.io.h5 import open_h5
from phy.io.kwik import create_kwik, KwikCreator, KwikModel
from phy.utils.event import ProgressReporter
from phy.traces.waveform import WaveformLoader, SpikeLoader
from phy.utils.logging import info


def _read_spikes(basename):
    with open_h5(basename + '.spiketimes.mat', 'r') as f:
        spike_samples = {}
        for name in f.children():
            cluster = int(name.split('_')[1])
            samples = f.read(name)[:].ravel().astype(np.uint64)
            spike_samples[cluster] = samples
        clusters = np.sort(list(spike_samples.keys()))
        # n_clusters = len(clusters)
        counts = {cluster: len(spikes)
                  for cluster, spikes in spike_samples.items()}
        spikes = np.hstack([spike_samples[cluster]
                            for cluster in clusters])
        idx = np.argsort(spikes)
        spike_clusters = np.repeat(clusters, [counts[cluster]
                                              for cluster in clusters])
        return spikes[idx], spike_clusters[idx]


def _read_templates(basename):
    with open_h5(basename + '.templates.mat', 'r') as f:
        templates = f.read('/templates')
        n_templates, n_samples, n_channels = templates.shape
        n_templates //= 2
        templates = templates[:n_templates, :, :]
        M = templates.max(axis=1).max(axis=1)
        templates /= M[:, None, None]
        masks = templates.max(axis=1)
    return templates, masks


def _truncate(fn, offset=None, n_channels=None, itemsize=None):
    """Eventually truncate a file at the end to ensure it has a correct shape.
    """
    n_bytes = os.stat(fn).st_size
    n_samples = (n_bytes - offset) // (itemsize * n_channels)
    n_bytes_final = offset + n_samples * n_channels * itemsize
    trunc = n_bytes - n_bytes_final
    assert n_bytes_final <= n_bytes
    assert trunc >= 0

    if trunc > 0:
        fn_copy = fn + '.dat'
        if op.exists(fn_copy):
            return (n_samples, n_channels)

        # Create the end-truncated file.
        info("Truncating...")
        shutil.copy(fn, fn_copy)
        with open(fn_copy, 'a') as f:
            f.truncate(n_bytes_final)
        assert os.stat(fn_copy).st_size == n_bytes_final <= n_bytes
        info("Truncated {} bytes at the end of `{}`.".format(trunc, fn))

    return (n_samples, n_channels)


def _read_filtered(basename, n_channels=None, dtype=None):
    fn = basename + '.filtered'
    with open(fn, 'rb') as f:
        data = f.read(4096)
    data = data.decode('ascii',  'ignore')
    i = data.index('EOH')
    # OFFSET = HEADER + EOH (3 bytes) + 2 uint16 samples (4 bytes)
    offset = i + 3 + 2 * 2
    info("Header: {} bytes.".format(offset))
    dtype = np.dtype(dtype)
    shape = _truncate(fn,
                      offset=offset,
                      n_channels=n_channels,
                      itemsize=dtype.itemsize)
    return np.memmap(fn + '.dat', dtype=dtype, offset=offset, shape=shape)


class Converter(object):
    def __init__(self,
                 basename,
                 n_channels=None,
                 prb_file=None,
                 dtype=None,
                 sample_rate=None,
                 ):

        self.n_features_per_channel = 3
        extract_s_before = extract_s_after = 30

        self.basename = basename
        self.kwik_path = basename + '.kwik'
        self.dtype = dtype
        self.prb_file = prb_file
        self.probe = load_probe(prb_file)

        self.sample_rate = sample_rate

        self._sd = SpikeDetekt(probe=self.probe,
                               n_features_per_channel=
                               self.n_features_per_channel,
                               pca_n_waveforms_max=10000,
                               extract_s_before=extract_s_before,
                               extract_s_after=extract_s_after,
                               sample_rate=sample_rate,
                               )
        self.n_samples_w = extract_s_before + extract_s_after

        # A xxx.filtered.trunc file may be created if needed.
        self.traces_f = _read_filtered(basename,
                                       n_channels=n_channels,
                                       dtype=dtype,
                                       )
        self.n_samples, self.n_channels = self.traces_f.shape
        assert n_channels == self.n_channels
        info("Loaded traces: {}.".format(self.traces_f.shape))

        # Load spikes.
        self.spike_samples, self.spike_clusters = _read_spikes(basename)
        self.n_spikes = len(self.spike_samples)
        assert len(self.spike_clusters) == self.n_spikes
        info("Loaded {} spikes.".format(self.n_spikes))

        # Chunks when computing features.
        self.chunk_size = 2500
        self.n_chunks = int(np.ceil(self.n_spikes / self.chunk_size))

        # Load templates and masks.
        self.templates, self.template_masks = _read_templates(basename)
        self.n_templates = len(self.templates)
        info("Loaded templates: {}.".format(self.templates.shape))

        # The WaveformLoader fetches waveforms from the raw traces dynamically.
        self._wl = WaveformLoader(traces=self.traces_f,
                                  n_samples=self.n_samples_w,
                                  dc_offset=32767.,
                                  scale_factor=.01,
                                  )
        # A virtual (n_spikes, n_samples, n_channels) array that is
        # memmapped to the filtered data file.
        self.waveforms = SpikeLoader(self._wl, self.spike_samples)

        assert self.waveforms.shape == (self.n_spikes,
                                        self.n_samples_w,
                                        self.n_channels)
        assert self.template_masks.shape == (self.n_templates, self.n_channels)

    def iter_spikes(self):
        for idx in range(0, self.n_chunks):
            i = idx * self.chunk_size
            j = (idx + 1) * self.chunk_size
            j_clip = min(j, self.n_spikes)
            yield (i, j_clip)

    def compute_pcs(self):
        k = self.n_spikes // self._sd._kwargs['pca_n_waveforms_max']

        # Find the masks of the selection of spikes.
        clu = self.spike_clusters[::k]
        masks = self.template_masks[clu]

        w, m = self.waveforms[::k], masks
        self.pcs = self._sd.waveform_pcs(w, m)
        return self.pcs

    def compute_features(self):
        pr = ProgressReporter()
        pr.set_progress_message('Computing features: {progress:.1f}%.')
        pr.set_complete_message('All features computed.')
        pr.value_max = self.n_chunks

        for i, j in self.iter_spikes():
            n = j - i

            # info("Extracting waveforms {} to {}...".format(i, j))
            w = self.waveforms[i:j]
            assert w.shape == (n, self.n_samples_w, self.n_channels)

            # info("Computing features of spikes {} to {}...".format(i, j))
            f = self._sd.features(w, self.pcs)
            assert f.shape == (n, self.n_channels, self.n_features_per_channel)

            yield f

            pr.increment()

    def compute_masks(self):
        for i, j in self.iter_spikes():
            n = j - i

            clu = self.spike_clusters[i:j]
            m = self.template_masks[clu]
            assert m.shape == (n, self.n_channels)

            yield m

    def create_kwik(self):
        # Create an empty Kwik file.
        info("Starting the conversion to Kwik...")
        create_kwik(kwik_path=self.kwik_path,
                    raw_data_files=[self.basename + '.filtered.dat'],
                    prb_file=self.prb_file,
                    n_channels=self.n_channels,
                    sample_rate=self.sample_rate,
                    dtype=self.dtype,
                    nfeatures_per_channel=self.n_features_per_channel,
                    overwrite=True,
                    )

        # Compute PCs and features.
        info("Computing PCs...")
        self.compute_pcs()

        info("Computing features of all spikes...")
        # WARNING: watch out RAM usage here. We cannot use a generator because
        # the KwiKCreator only accepts lists at the moment.
        features = (f for f in self.compute_features())
        masks = (m for m in self.compute_masks())

        # Add spikes.
        info("Adding the spikes in the kwik file.")
        creator = KwikCreator(self.kwik_path)
        creator.add_spikes(group=1,
                           spike_samples=self.spike_samples,
                           masks=masks,
                           features=features,
                           )

        # Add clusters.
        info("Adding the clusters in the kwik file.")
        creator.add_clustering(group=1,
                               name='main',
                               spike_clusters=self.spike_clusters,
                               )

        info("Kwik file successfully created!")

    def template_explorer(self, name='templates'):
        """Mini GUI to explore the templates."""

        from phy.plot.waveforms import plot_waveforms
        from vispy.app import run

        p = c.probe['channel_groups'][1]['geometry']
        positions = [p[channel] for channel in sorted(p)]

        self._n = 2
        wave = np.zeros((0, self.n_samples_w, self.n_channels))
        w = plot_waveforms(channel_positions=positions,
                           waveforms=wave,
                           overlap=True,
                           alpha=1.,
                           probe_scale=(1.9, 1.0),
                           box_scale=(0.066, 0.01),
                           )

        # Show templates.
        if name == 'templates':
            templates = self.templates
            masks = self.template_masks

        # Show waveforms.
        elif name == 'waveforms':
            templates = self.waveforms
            masks = self.template_masks[self.spike_clusters]

        @w.connect
        def on_key_press(e):
            if e.key == 'space':
                self._n += 1 if ('Shift' not in e.modifiers) else -1
                if name == 'templates':
                    info("Template {}.".format(self._n))
                    w.set_data(waveforms=templates[self._n],
                               masks=masks[self._n],
                               )
                elif name == 'waveforms':
                    sample = self.spike_samples[self._n]
                    cluster = self.spike_clusters[self._n]
                    info("Waveform {}, template={}, sample={}.".format(self._n,
                         cluster, sample))
                    wav = np.vstack((templates[self._n],
                                     self.templates[cluster][:-1][None, ...]))
                    m = np.vstack((masks[self._n],
                                   self.template_masks[cluster][None, ...]))
                    w.set_data(waveforms=wav,
                               masks=m,
                               spike_clusters=[0, 1],
                               )
        run()


if __name__ == '__main__':

    basename = 'checkerboard'
    prb_file = 'mea_252.prb'
    n_channels = 252
    sample_rate = 25000
    dtype = 'uint16'

    c = Converter(basename,
                  n_channels=n_channels,
                  prb_file=prb_file,
                  sample_rate=sample_rate,
                  dtype=dtype,
                  )

    # Uncomment to have a look at the templates or waveforms.
    # c.template_explorer('waveforms')  # 'waveforms' or 'templates'
    # exit()

    if not os.path.exists(basename + '.kwik'):
        # Conversion.
        c.create_kwik()

    # Try to open the kwik file after the conversion.
    model = KwikModel(c.kwik_path)
    model.describe()
