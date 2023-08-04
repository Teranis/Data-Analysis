import core
import OD
import CC
import spotMAXana
import CellACDCana
import matplotlib
print('Running Data Analysis')
matplotlib.use('TkAgg')
matplotlib.rc('axes', axisbelow=True)
OD.odplot()
####Possible commands
##CC
#CC.coultercounter()
#CC.plotfitdata()
#CC.boxplot()
##OD
#OD.odplot()
#OD.doublingtime()
##spotMAXana
#spotMAXana.boxplot()
##CellACDCana
#CellACDCana.checkingcompleteness()
######Don't forget to change the path and other parameters in the respective config files