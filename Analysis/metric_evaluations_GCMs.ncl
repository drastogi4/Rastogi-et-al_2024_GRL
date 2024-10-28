load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin
;Experiments
 experiments = (/"SRCNN-MSE","SRCNN-MSE-EL","SRCNN-MSE-EL-Proc","SRCNN-EXP-EL-Proc","SRCNN-QT1-EL-Proc","SRCNN-QT2-EL-Proc"/)

 gcms        = (/"ACCESS-CM2","BCC-CSM2-MR","CNRM-ESM2-1","MPI-ESM1-2-HR","MRI-ESM2-0","NorESM2-MM"/)

;Path to monthly WRF output, ERA5 data and SRCNN downscaled GCMs
 dir         = "./analysis/"
 varin       = "avg"
 timper      = "1980-2019"
 sim         = "gcm_predict"

;Read ERA5-DD/WRF output
 fw          = addfile(dir+"WRF_"+varin+".nc","r")
 RAIN        = fw->$var$

;Read ERA5 data
 ff1         = addfile(dir+"ERA5_"+varin+".nc","f")
 RAINe       = ff1->$var$

 RAINf1      = new((/dimsizes(gcms),dimsizes(experiments),216,488/),typeof(RAIN))
 do gg       = 0, dimsizes(gcms)-1
 gcm         = gcms(gg)
 do ee       = 0,dimsizes(experiments)-1

 exper       = experiments(ee)
 ff1         = addfile(dir+"pr_"+gcm+"_predict_daily_"+exper+"_"+varin+".nc","r")
 RAINf1(gg,ee,:,:) = ff1->pr(0,:,:)

 end do
 end do

 RAINf      = dim_avg_n_Wrap(RAINf1,0)

;Read grid information
  lat       = fw->lat
  lon       = fw->lon
  RAIN&lat  = lat
  RAIN&lon  = lon
  RAINf1!2  = "lat"
  RAINf2!3  = "lon"
  RAINf&lat = lat1
  RAINf&lon = lon1

;Plotting
  pdfname=varin+"_avgGCM_"+timper+"_"+sim
  wks  = gsn_open_wks ("pdf",pdfname)
  setvalues NhlGetWorkspaceObjectId()
  "wsMaximumSize" : 300000000
  end setvalues
  plot = new ( 8, "graphic" )
  gsn_define_colormap (wks, "BlAqGrYeOrRevi200"); Temperature
  res                        = True               ; plot mods desired
  cmap                       = gsn_retrieve_colormap(wks)
  cmap(2:,:)                 = cmap(2::-1,:)
  gsn_define_colormap(wks,cmap)
  res@gsnDraw                = False
  res@gsnFrame               = False
  res@cnRasterModeOn         = True
  res@cnFillOn               = True
  res@cnLinesOn              = False
  res@cnLineLabelsOn         = False
  res@gsnSpreadColors        = True               ; use total colormap
  res@cnInfoLabelOn          = False
  res@tiMainFontHeightF      = 0.020
  res@gsnAddCyclic           = False
  res@tiMainOffsetYF         = 0.
  res@gsnTickMarksOn         = False
  res@mpGridAndLimbOn        = False
  res@cnLevelSelectionMode   = "ExplicitLevels"
  res@cnLevelSelectionMode   = "AutomaticLevels"
  res@cnLevelSelectionMode   = "ManualLevels"
  res@mpPerimOn              = False
  res@mpLandFillColor        = "white"
  res@mpOutlineBoundarySets  = "USStates"
  res@mpUSStateLineColor     = "Black"
  res@mpNationalLineColor    = "Black"
  res@mpGeophysicalLineColor = "Black"
  res@mpUSStateLineThicknessF= 0.01
  ilon                       = dimsizes(lon)
  jlat                       = dimsizes(lat)
  res@mpLimitMode            = "Corners"            ; choose range of map
  res@mpLeftCornerLatF       = lat(0)
  res@mpLeftCornerLonF       = lon(0)
  res@mpRightCornerLatF      = lat(jlat-1)
  res@mpRightCornerLonF      = lon(ilon-1)
  res@cnLevelSpacingF        = 0.5;0
  res@cnMinLevelValF         = 0
  res@cnMaxLevelValF         = 5
  res@lbLabelStride          = 2

  res@tiMainString = "ERA "
  plot(0)          = gsn_csm_contour_map(wks,tp,res)

  res@tiMainString = "WRF "
  plot(1)          = gsn_csm_contour_map(wks,RAIN,res)

  do ee            = 0, dimsizes(experiments)-1

  res@tiMainString = "GCM "+timper+" "+experiments(ee)
  idx              = ee+3
  RAINf11          = RAINf(ee,:,:)
  plot(idx)        = gsn_csm_contour_map(wks,RAINf11,res)

  end do
  pres             = True
  pres@gsnPanelMainString = varin
  gsn_panel(wks,(/plot/),(/4,2/),pres)
end
