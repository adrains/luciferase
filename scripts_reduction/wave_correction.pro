function model_scale,x,data,model,POWER=power

  a=dblarr(power+1,power+1)
  b=dblarr(power+1)
  for i=0,power do begin
    for j=0,power do begin
      a[j,i]=total(model*model*x^(i+j))
    endfor
    b[i]=total(data*model*x^i)
  endfor
  LA_LUDC, a, index
  coef = LA_LUSOL(a, index, b)

  return, poly(x, coef)
end

function lmfunct, par, dpar, ERR=err, fa=fa
  common data,wo,so,so2,w_ref,s_ref,x,weight,debug_plot

  p=par
  p[0]=p[0]+1.d0

; x is a vector ranging from -1 to +1 with the same number of elements as the reference wavelengths
  w = poly(x,p)*w_ref
  model = so
  i1=(where(w ge wo[0]))[0]
  i2=(where(w le wo[n_elements(wo)-1], ni2))[ni2-1]
  model[i1:i2] = spl_interp(wo,so,so2,w[i1:i2],/DOUBLE)

  c=model_scale(x,s_ref,model,POWER=4)
  model=model*c

  if keyword_set(debug_plot) then begin
    plot,w_ref,s_ref,xs=3 & oplot,w_ref,model,col=c24(2)
    oplot,w_ref,(s_ref-model)+0.5*mean(model),line=1
    print,sqrt(total((s_ref-model)^2)),par
    wait,0.1
;    stop
  endif

  return,(s_ref-model)*weight
end

function wl_morph,bary,wl_sci,sp_sci,unc_sci,wl_ref,sp_ref $
                 ,wl_sta,sp_sta,POWER=power,POLY=poly_coef $
                 ,WEIGHTS=weights,STAT=stat,PLOT=plot

  common data,wo,so,so2,w_ref,s_ref,x,weight,debug_plot

; VERIFY INPUT DATA
  if n_elements(power) eq 0 then power=2
  if n_elements(wl_sci) ne n_elements(sp_sci) then begin
    print,'Number of wavelength does not match the size of the spectrum for science.'
    return,-1
  endif

  npix=n_elements(wl_sci)
  if n_elements(wl_tell) ne n_elements(sp_tell) then begin
    print,'Number of wavelength does not match the size of the spectrum for tellurics.'
    return,-2
  endif

  if keyword_set(weights) then weight=weights $
  else                         weight=replicate(1.d0, npix)

  if keyword_set(plot) then debug_plot=1 else debug_plot=0

; -------------------------------------------------------------		
; ------------- CALIBRATION WITH TELLURIC SPECTRA -------------
; -------------------------------------------------------------		

  x=(dindgen(npix)/npix*2.d0-1.d0)

; Initial guess vector package according to MPFIT rules

  parinfo = replicate({value:0.D, fixed:0, limited:[1,1], $
                      limits:[0.D,0]}, power+1)

  for ipar=0,power do begin ; Setting limits on polynomila coefficients.
                            ; Limits are the same for all coefficients
                            ; because of normalisation of x above
    parinfo[ipar].limits=[-1.d-3,1.d-3]
  endfor

  w_ref=wl_ref
  s_ref=sp_ref
  wo=wl_sci
  so=sp_sci
  so2=spl_init(wo,so,/DOUBLE)

; Initial guess for the morphing parameters

  a=dblarr(power+1) & a[0]=1.d0

  fa={x:wo, y:so, err:sqrt(abs(so))}

  a=mpfit('lmfunct',a,functargs=fa,PARINFO=parinfo,maxiter=500,STATUS=stat,/AUTODERIV,/QUIET)

  poly_coef=a[0:power]
  p=poly_coef
  p[0]=p[0]+1.d0
  w_out=poly(x,p)*w_ref

  return,w_out
end

pro wave_correction,wave,obs,unc,wave_ref,obs_ref,POLY_COEF=poly_co $
                   ,WEIGHTS=weights,POWER=power,MORPH=morph,DEBUG=debug
;
; Wavelength morphing tool.
;   Inputs: wave     - wavelength vector of a spectrum to be morphed
;           obs      - observations
;           unc      - uncertainty
;           wave_ref - wavelength vector of the refence spectrum (to be matched)
;           obs_ref  - reference spectrum
;           weights  - pixel weights set on the reference wavelength (0-1, 0 - ignore)
;           power    - power of the polynomial (default 2)
;  Outputs: poly_co  - best fitting polynomial coefficients. The transformation is then:
;                      x = (dindgen(npix)/npix*2.d0-1.d0)
;                      p = poly_co
;                      p[0] = p[0] + 1
;                      w = poly(x,p) * wave_ref
;                      so2 = spl_init(wave, obs, /DOUBLE)
;                      obs_new = spl_interp(wave, obs, so2, w, /DOUBLE)
;
; Switches: /MORPH   - Morphing procedure is performed by wave_correction and the new spectrum
;                      set on the wave_ref scale is returned as obs. In this case there is also
;                      provision to avoid extrapolation. Without this switch onle the
;                      the coefficients for wavelength transofrmation are retirned without
;                      modification to obs.
;           /DEBUG   - turns on debug plotting.
;
;  Written by N. Piskunov, July 2025.

  nwave=n_elements(wave[*,0])

  if n_elements(power) eq 0 then power=2

  w_ref=wave_ref
  wb=wave
  j1=min(where(wb ge w_ref[0]))
  j2=max(where(wb le w_ref[n_elements(w_ref)-1]))
  s_ref=obs_ref
  weight=abs(1.1d0-s_ref)

  sb=obs

  x=(dindgen(npix)/npix*2.d0-1.d0)
  rb=model_scale(x,sb,s_ref,POWER=4)
  sb=sb/rb

  sb2=spl_init(wb,sb,/DOUBLE)

  wb_new=wl_morph(0.d0,wb,sb,unc,w_ref,s_ref,POLY=poly_co,WEIGHTS=weights $
                      ,POWER=power,STAT=stat,PLOT=debug)
  if keyword_set(debug) then begin
    plot,w_ref,s_ref/median(s_ref),xs=1
    rb_new=spl_interp(wb,sb,sb2,wb_new,/DOUBLE)
    oplot,w_ref,rb_new/median(rb_new),col=c24(2)
  endif

  if keyword_set(morph) then begin
    sb_new=sb
    k1=min(where(wb_new gt wb[0]))
    k2=max(where(wb_new lt wb[nwave-1]))
    sb_new[k1:k2]=spl_interp(wb,sb,sb2,wb_new[k1:k2],/DOUBLE)
    rb_new=rb
    rb2=spl_init(wb,rb,/DOUBLE)
    rb_new[k1:k2]=spl_interp(wb,rb,rb2,wb_new[k1:k2],/DOUBLE)
    unc2=spl_init(wb,unc,/DOUBLE)
    unc[k1:k2]=spl_interp(wb,unc,unc2,wb_new[k1:k2],/DOUBLE)
    obs=sb_new*rb_new
  endif

end
