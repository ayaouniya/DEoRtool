import numpy as np
import healpy as hp
from scipy.interpolate import interp1d, pchip
import h5py
from astropy import units
from component_data import GSM_FILEPATH
import ephem
from astropy.time import Time

#基础全天模型
class LGSM(ephem.Observer):
    
    #set
    
    def __init__(self,filepath=GSM_FILEPATH,freq_unit='MHz',data_unit='K',basemap='haslam',interpolation='pchip'):
        
        self.name = 'GSM2008'
        self.filepath = filepath
        self.freq_unit = freq_unit
        self.data_unit = data_unit
        self.basemap = basemap
        self.interpolation_method = interpolation
        
        self.h5 = h5py.File(filepath,"r")
        
        self.pca_map_data = None
        self.interp_comps = None
        self.update_interpolants()

        self.generated_map_data = None
        self.generated_map_freqs = None
        
        self.nside = 512
        self.dec = False
        
    def setup(self):
        
        self._n_side = 512
        self._n_pix = 3145728
        self._theta, self._phi = hp.pix2ang(self._n_side, np.arange(self._n_pix))
        self._time = Time(self.date.datetime())
        self._pix0 = None
        self._mask = None
        
    def update_interpolants(self):
        
        pca_map_dict = {"5deg": "component_maps_5deg",
                        "haslam": "component_maps_408locked",
                        "wmap": "component_maps_23klocked"}
        pca_map_key = pca_map_dict[self.basemap]
        self.pca_map_data = self.h5[pca_map_key][:]

        pca_table = self.h5["components"][:]
        pca_freqs_mhz = pca_table[:, 0]
        pca_scaling   = pca_table[:, 1]
        pca_comps     = pca_table[:, 2:].T

        ln_pca_freqs = np.log(pca_freqs_mhz)

        if self.interpolation_method == 'cubic':
            spl_scaling = interp1d(ln_pca_freqs, np.log(pca_scaling), kind='cubic')
            spl1 = interp1d(ln_pca_freqs,   pca_comps[0],   kind='cubic')
            spl2 = interp1d(ln_pca_freqs,   pca_comps[1],   kind='cubic')
            spl3 = interp1d(ln_pca_freqs,   pca_comps[2],   kind='cubic')

        else:
            spl_scaling = pchip(ln_pca_freqs, np.log(pca_scaling))
            spl1 = pchip(ln_pca_freqs,   pca_comps[0])
            spl2 = pchip(ln_pca_freqs,   pca_comps[1])
            spl3 = pchip(ln_pca_freqs,   pca_comps[2])
        self.interp_comps = (spl_scaling, spl1, spl2, spl3)       
    
    def set_basemap(self, new_basemap):
        self.basemap = new_basemap
        self.update_interpolants()
        if self.generated_map_freqs is not None:
            self.generate(self.generated_map_freqs)

    def set_freq_unit(self, new_unit):
        self.freq_unit = new_unit
        self.update_interpolants()
        if self.generated_map_freqs is not None:
            self.generate(self.generated_map_freqs)

    def set_interpolation_method(self, new_method):
        self.interpolation_method = new_method
        self.update_interpolants()
        if self.generated_map_freqs is not None:
            self.generate(self.generated_map_freqs)

    #generate and mask

    def generate(self,freqs):
        self.freq = freqs
        freqs = np.array(freqs) * units.Unit(self.freq_unit)
        freqs_mhz = freqs.to('MHz').value

        if isinstance(freqs_mhz, float):
            freqs_mhz = np.array([freqs_mhz])

        try:
            assert np.min(freqs_mhz) >= 10
            assert np.max(freqs_mhz) <= 94000
        except AssertionError:
            raise RuntimeError("Frequency values lie outside 10 MHz < f < 94 GHz")

        ln_freqs     = np.log(freqs_mhz)
        spl_scaling, spl1, spl2, spl3 = self.interp_comps
        comps = np.row_stack((spl1(ln_freqs), spl2(ln_freqs), spl3(ln_freqs)))
        scaling = np.exp(spl_scaling(ln_freqs))
        
        map_out = np.einsum('cf,pc,f->fp', comps, self.pca_map_data, scaling)

        if map_out.shape[0] == 1:
            map_out = map_out[0]
        self.generated_map_data = map_out
        self.generated_map_freqs = freqs
        
        return map_out
    
    def dec_res(self,n_side):
        
        gmap = self.generated_map_data
        lower_res = hp.ud_grade(gmap,n_side)
        self.generated_map_data = lower_res
        self._n_side = n_side
        self._n_pix = hp.nside2npix(self._n_side)
        self._theta, self._phi = hp.pix2ang(self._n_side, np.arange(self._n_pix))
        self._time = Time(self.date.datetime())
        self._pix0 = None
        self._mask = None
        self.dec = True
        
        return lower_res

    def mask_sky(self,idx=0,obstime=None):
        
        self.setup()
        
        if self.generated_map_data.ndim == 2:
            gmap = self.generated_map_data[idx]
            freq = self.generated_map_freqs[idx]
        else:
            gmap = self.generated_map_data
            freq = self.generated_map_freqs
        
        self.f = freq
        
        if obstime == self._time or obstime is None:
            time_has_changed = False
        else:
            time_has_changed = True
            self._time = Time(obstime)  
            self.date  = obstime.to_datetime()
            
        ra_rad, dec_rad = self.radec_of(0, np.pi/2)
        ra_deg  = ra_rad / np.pi * 180
        dec_deg = dec_rad / np.pi * 180

        hrot = hp.Rotator(rot=[ra_deg, dec_deg], coord=['G', 'C'], inv=True)
        g0, g1 = hrot(self._theta, self._phi)
        pix0 = hp.ang2pix(self._n_side, g0, g1)

        mask1 = self._phi + np.pi / 2 > 2 * np.pi
        mask2 = self._phi < np.pi / 2
        mask = np.invert(np.logical_or(mask1, mask2))
        self._pix0 = pix0
        self._mask = mask

        sky = gmap
        sky_rotated = sky[self._pix0]
        self.observed_sky = hp.ma(sky_rotated)
        self.observed_sky.mask = self._mask
        
        return self.observed_sky
    
    def masks_sky(self, n_side=512, dec_res = False, obstime=None):
        
        if self.dec==False:
            self.setup()

        gmap = self.generated_map_data
        freq = self.generated_map_freqs
        
        self.f = freq
        
        if obstime == self._time or obstime is None:
            time_has_changed = False
        else:
            time_has_changed = True
            self._time = Time(obstime)  
            self.date  = obstime.to_datetime()
            
        ra_rad, dec_rad = self.radec_of(0, np.pi/2)
        ra_deg  = ra_rad / np.pi * 180
        dec_deg = dec_rad / np.pi * 180

        hrot = hp.Rotator(rot=[ra_deg, dec_deg], coord=['G', 'C'], inv=True)
        g0, g1 = hrot(self._theta, self._phi)
        pix0 = hp.ang2pix(self._n_side, g0, g1)

        mask1 = self._phi + np.pi / 2 > 2 * np.pi
        mask2 = self._phi < np.pi / 2
        mask = np.invert(np.logical_or(mask1, mask2))
        self._pix0 = pix0
        self._mask = mask

        sky = gmap
        sky_rotated = sky[:,self._pix0]
        self.observed_sky = hp.ma(sky_rotated)
        self.observed_sky.mask = self._mask
        
        return self.observed_sky

    #view
    
    def view(self, idx=0, logged=False, show=False):

        if self.generated_map_data is None:
            raise RuntimeError("No GSM map has been generated yet. Run generate() first.")

        if self.generated_map_data.ndim == 2:
            gmap = self.generated_map_data[idx]
            freq = self.generated_map_freqs[idx]
        else:
            gmap = self.generated_map_data
            freq = self.generated_map_freqs

        if logged:
            gmap = np.log2(gmap)

        hp.mollview(gmap, coord='G', title='%s %s, %s' % (self.name, str(freq), self.basemap))
        
    def view_observed_sky(self,idx=0,logged=True):
        
        if self.observed_sky.ndim == 2:
            sky = self.observed_sky[idx]
            freq = self.generated_map_freqs[idx]
        else:
            sky = self.observed_sky
            freq = self.generated_map_freqs
               
        if logged:
            sky = np.log2(sky)

        hp.orthview(sky,half_sky=True,title='%s %s %s %s' %(freq,self._time,self.lon,self.lat))
        
    def view_observed_g(self, logged=False, show=False, **kwargs):

        sky = self.observed_sky
        if logged:
            sky = np.log2(sky)

        ra_rad, dec_rad = self.radec_of(0, np.pi / 2)
        ra_deg  = ra_rad / np.pi * 180
        dec_deg = dec_rad / np.pi * 180

        derotate = hp.Rotator(rot=[ra_deg, dec_deg])
        g0, g1 = derotate(self._theta, self._phi)
        pix0 = hp.ang2pix(self._n_side, g0, g1)
        sky = sky[pix0]

        coordrotate = hp.Rotator(coord=['C', 'G'], inv=True)
        g0, g1 = coordrotate(self._theta, self._phi)
        pix0 = hp.ang2pix(self._n_side, g0, g1)
        sky = sky[pix0]

        hp.mollview(sky, coord='G', title='Observed Sky in Galacctic',**kwargs)

        return sky
    
    def view_observed_c(self, logged=False, show=False, **kwargs):

        sky = self.observed_sky
        if logged:
            sky = np.log2(sky)

        ra_rad, dec_rad = self.radec_of(0, np.pi / 2)
        ra_deg  = ra_rad / np.pi * 180
        dec_deg = dec_rad / np.pi * 180

        derotate = hp.Rotator(rot=[ra_deg, dec_deg])
        g0, g1 = derotate(self._theta, self._phi)
        pix0 = hp.ang2pix(self._n_side, g0, g1)
        sky = sky[pix0]

        hp.mollview(sky, coord='C', title = 'Observed Sky in Equatorial',**kwargs)
        
        return sky
    
    
