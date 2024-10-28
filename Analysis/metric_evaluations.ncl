load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin
;Experiments
 experiments = (/"SRCNN-MSE","SRCNN-MSE-EL","SRCNN-MSE-EL-Proc","SRCNN-EXP-EL-Proc","SRCNN-QT1-EL-Proc","SRCNN-QT2-EL-Proc"/)

;Path to WRF output and ERA5 data and SRCNN output
 dir       = "./analysis/"
 varin     = "avg"
 var       = "pr"
 year      = ispan(1981,2019,1)
 timper    = "2010-2019"

;Read ERA5-DD/WRF output
 fw        = addfile(dir+"WRF_"+varin+".nc","r")
 RAIN      = fw->$var$

;Read ERA5 data
 ff1       = addfile(dir+"ERA5_"+varin+".nc","f")
 RAINe     = ff1->$var$
 RAINee    = RAINe-RAIN
 copy_VarCoords(RAIN, RAINe)
 dims      = dimsizes(RAIN)
 RAINss    = new(/dimsizes(experiments),dims(0),dims(1)/),"float")

;Read SRCNN downscaled output
 do ee     = 0, dimsizes(experiments)-1
  ffe           = addfile(dir+"y_"+timeper+"_predict_daily_"+experiments(ee)+"_"+varin+".nc","r")
  RAINs         = ff2->$var$
  RAINss(ee,:,:)= RAINs-RAIN
 end do
;Read grid information
  lat = fw->lat
  lon = fw->lon
  RAINss!1 = "lat"
  RAINss!2 = "lon"
  RAINss&lat = lat
  RAINss&lon = lon
  pdfname                     = var+"_"+varin+"_all_"+timper+"_diff_final"
  wks                         = gsn_open_wks ("pdf",pdfname)
  setvalues NhlGetWorkspaceObjectId()
  "wsMaximumSize" : 300000000
  end setvalues
  plot                        = new ( 8, "graphic" )
  gsn_define_colormap (wks,"BlRe")
  res                         = True               ; plot mods desired
  cmap                        = gsn_retrieve_colormap(wks)
  cmap(2:,:)                  = cmap(2::-1,:)
  gsn_define_colormap(wks,cmap)
  res@gsnDraw                 = False
  res@gsnFrame                = False
  res@cnRasterModeOn          = True
  res@cnFillOn                = True
  res@cnLinesOn               = False
  res@cnLineLabelsOn          = False
  res@gsnSpreadColors         = True               ; use total colormap
  res@cnInfoLabelOn           = False
  res@tiMainFontHeightF       = 0.020
  res@gsnAddCyclic            = False
  res@tiMainOffsetYF          = 0.
  res@gsnTickMarksOn          = False
  res@mpGridAndLimbOn         = False
  res@cnLevelSelectionMode    = "ExplicitLevels"
  res@cnLevelSelectionMode    = "AutomaticLevels"
  res@cnLevelSelectionMode    = "ManualLevels"
  res@mpPerimOn               = False
  res@mpOutlineBoundarySets   = "USStates"
  res@mpUSStateLineColor      = "Black"
  res@mpNationalLineColor     = "Black"
  res@mpGeophysicalLineColor  = "Black"
  res@mpUSStateLineThicknessF = 0.01
  ilon                        = dimsizes(lon)
  jlat                        = dimsizes(lat)
  res@mpLimitMode             = "Corners"            ; choose range of map
  res@mpLeftCornerLatF        = lat(0)
  res@mpLeftCornerLonF        = lon(0)
  res@mpRightCornerLatF       = lat(jlat-1)
  res@mpRightCornerLonF       = lon(ilon-1)
  res@cnLevelSpacingF         = 0.5
  res@cnMinLevelValF          = -5
  res@cnMaxLevelValF          = 5
  res@lbLabelStride           = 2
  res@tiMainString            = "ERA5"
  plot(0)                     = gsn_csm_contour_map(wks,RAINaa,res)
  do ee                       = 0, dimsizes(experiments)-1
  res@tiMainString            = exper(ee)
  plot(ee+1)                  = gsn_csm_contour_map(wks,RAINss(ee,:,:),res)
  end do
  pres                        = True
  pres@gsnPanelMainString     = varin +" Differences with respect to ERA5-DD"
  gsn_panel(wks,(/plot/),(/4,2/),pres)
end
