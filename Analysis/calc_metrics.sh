#!/bin/csh -f
  
set inputdir  = './analysis/'
set outputdir = './analysis/'

foreach ee ( SRCNN-MSE SRCNN-MSE-EL SRCNN-MSE-EL-Proc SRCNN-EXP-EL-Proc SRCNN-QT-EL-Proc )
foreach tt (2010-2019)
set v = 'y_'$tt'_predict_daily_'$ee
cdo selvar,pr $inputdir$v'.nc' temp.nc
cdo setrtomiss,0,1.0 temp.nc pre.nc
cdo timmean temp.nc $inputdir$v'_avg.nc'
cdo eca_rr1,1.0 pre.nc $inputdir$v'_wetdays.nc'
cdo eca_rr1,25.0 pre.nc $inputdir$v'_25mmdays.nc'
# Calculate percentile
cdo yearmin temp.nc  min.nc
cdo yearmax temp.nc max.nc
cdo yearpctl,95 temp.nc min.nc max.nc temp95.nc
cdo timmean temp95.nc $v'_p95.nc'

#Calculate precipitation from extremes
rm temp.nc
rm pre.nc
rm min.nc
rm max.nc
rm temp95.nc
end
end
