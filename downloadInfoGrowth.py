import yfinance as yf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from yahoo_fin import stock_info as si
import keras
import time

tickers = ['A',
           'AA',
           'AAN',
           'AAP',
           'AAT',
           'AB',
           'ABB',
           'ABBV',
           'ABC',
           'ABEV',
           'ABG',
           'ABM',
           'ABR',
           'ABT',
           'AC',
           'ACA',
           'ACB',
           'ACC',
           'ACCO',
           'ACEL',
           'ACH',
           'ACM',
           'ACN',
           'ACP', 'ACRE', 'ACV', 'ADC', 'ADM', 'ADNT', 'ADS', 'ADSW', 'ADT', 'ADX', 'AEB', 'AEE', 'AEFC', 'AEG', 'AEL', 'AEM', 'AEO', 'AEP', 'AER', 'AES', 'AFB', 'AFC', 'AFG', 'AFGH', 'AFI', 'AFL', 'AFT', 'AG', 'AGCO', 'AGD', 'AGI', 'AGM', 'AGN', 'AGO', 'AGR', 'AGRO', 'AGS', 'AGX', 'AHC', 'AHH', 'AHT', 'AI', 'AIC', 'AIF', 'AIG', 'AIN', 'AIO', 'AIR', 'AIT', 'AIV', 'AIW', 'AIZ', 'AJG', 'AJRD', 'AJX', 'AKO.A', 'AKO.B', 'AKR', 'AL', 'ALB', 'ALC', 'ALE', 'ALEX', 'ALG', 'ALI-A', 'ALI-B', 'ALI-E', 'ALK', 'ALL', 'ALL-I', 'ALLE', 'ALLY', 'ALSN', 'ALTG', 'ALUS', 'ALV', 'ALX', 'AM', 'AMB.W', 'AMBC', 'AMC', 'AMCR', 'AME', 'AMG', 'AMH', 'AMK', 'AMN', 'AMOV', 'AMP', 'AMPY', 'AMRC', 'AMRX', 'AMT', 'AMX', 'AN', 'ANET', 'ANF', 'ANH', 'ANTM', 'AOD', 'AON', 'AP', 'APA', 'APAM', 'APD', 'APG', 'APH', 'APHA', 'APLE', 'APO', 'APRN', 'APTS', 'APTV', 'APY', 'AQN', 'AQNA', 'AQUA', 'AR', 'ARA', 'ARC', 'ARCH', 'ARCO', 'ARD', 'ARDC', 'ARE', 'ARES', 'ARGD', 'ARGO', 'ARI', 'ARL', 'ARLO', 'ARMK', 'ARNC', 'AROC', 'ARR', 'ARW', 'ASA', 'ASB', 'ASC', 'ASG', 'ASGN', 'ASH', 'ASIX', 'ASPN', 'ASR', 'ASX', 'AT', 'ATCO', 'ATEN', 'ATGE', 'ATH', 'ATH-B', 'ATHM', 'ATI', 'ATKR', 'ATO', 'ATR', 'ATTO', 'ATUS', 'ATV', 'AU', 'AUY', 'AVA', 'AVAL', 'AVB', 'AVD', 'AVH', 'AVK', 'AVLR', 'AVNS', 'AVTR', 'AVY', 'AVYA', 'AWI', 'AWK', 'AWP', 'AWR', 'AX', 'AXE', 'AXL', 'AXO', 'AXP', 'AXR', 'AXS', 'AXTA', 'AYI', 'AYX', 'AZN', 'AZO', 'AZRE', 'AZUL', 'AZZ', 'B', 'BA', 'BABA', 'BAC', 'BAC-N', 'BAF', 'BAH', 'BAK', 'BAM', 'BANC', 'BAP', 'BAX', 'BB', 'BBAR', 'BBD', 'BBDC', 'BBDO', 'BBF', 'BBK', 'BBL', 'BBN', 'BBU', 'BBVA', 'BBW', 'BBX', 'BBY', 'BC', 'BCC', 'BCE', 'BCEI', 'BCH', 'BCO', 'BCS', 'BCSF', 'BCX', 'BDC', 'BDJ', 'BDN', 'BDX', 'BDXA', 'BE', 'BEDU', 'BEN', 'BEP', 'BERY', 'BEST', 'BFAM', 'BFK', 'BFO', 'BFS', 'BFY', 'BFZ', 'BG', 'BGB', 'BGG', 'BGH', 'BGIO', 'BGR', 'BGS', 'BGSF', 'BGT', 'BGX', 'BGY', 'BH', 'BH.A', 'BHC', 'BHE', 'BHK', 'BHLB', 'BHP', 'BHR', 'BHV', 'BHVN', 'BIF', 'BIG', 'BILL', 'BIO', 'BIP', 'BIPC', 'BIT', 'BITA', 'BJ', 'BK', 'BKD', 'BKE', 'BKH', 'BKI', 'BKK', 'BKN', 'BKR', 'BKT', 'BKU', 'BLD', 'BLE', 'BLK', 'BLL', 'BLW', 'BLX', 'BMA', 'BME', 'BMEZ', 'BMI', 'BMO', 'BMY', 'BNED', 'BNS', 'BNY', 'BOE', 'BOH', 'BOOT', 'BORR', 'BOX', 'BP', 'BPMP', 'BPT', 'BQH', 'BR', 'BRBR', 'BRC', 'BRFS', 'BRMK', 'BRO', 'BRT', 'BRX', 'BSA', 'BSAC', 'BSBR', 'BSD', 'BSE', 'BSIG', 'BSL', 'BSM', 'BSMX', 'BST', 'BSTZ', 'BSX', 'BTA', 'BTE', 'BTI', 'BTO', 'BTT', 'BTU', 'BTZ', 'BUD', 'BUI', 'BURL', 'BV', 'BVN', 'BW', 'BWA', 'BWG', 'BWXT', 'BX', 'BXC', 'BXG', 'BXMT', 'BXMX', 'BXP', 'BXS', 'BY', 'BYD', 'BYM', 'BZH', 'BZM', 'C', 'CAAP', 'CABO', 'CACI', 'CADE', 'CAE', 'CAF', 'CAG', 'CAH', 'CAI', 'CAJ', 'CAL', 'CALX', 'CANG', 'CAPL', 'CARS', 'CAT', 'CATO', 'CB', 'CBB', 'CBD', 'CBH', 'CBL', 'CBRE', 'CBT', 'CBU', 'CBZ', 'CC', 'CCAC', 'CCC', 'CCEP', 'CCH', 'CCH.U', 'CCI', 'CCJ', 'CCK', 'CCL', 'CCM', 'CCO', 'CCR', 'CCS', 'CCU', 'CCX', 'CCX.U', 'CCXX', 'CCZ', 'CDAY', 'CDE', 'CDR', 'CE', 'CEA', 'CEE', 'CEIX', 'CEL', 'CELP', 'CEM', 'CEN', 'CEO', 'CEPU', 'CEQP', 'CF', 'CFG', 'CFG-E', 'CFR', 'CFX', 'CFXA', 'CGA', 'CGC', 'CHA', 'CHAP', 'CHCT', 'CHD', 'CHE', 'CHGG', 'CHH', 'CHK', 'CHL', 'CHMI', 'CHN', 'CHRA', 'CHS', 'CHT', 'CHU', 'CHWY', 'CI', 'CIA', 'CIB', 'CIEN', 'CIF', 'CIG', 'CII', 'CIM', 'CINR', 'CIO', 'CIR', 'CIT', 'CKH', 'CL', 'CLB', 'CLDR', 'CLDT', 'CLF', 'CLGX', 'CLH', 'CLI', 'CLNC', 'CLNY', 'CLPR', 'CLR', 'CLS', 'CLW', 'CLX', 'CM', 'CMA', 'CMC', 'CMCM', 'CMD', 'CMG', 'CMI', 'CMO', 'CMP', 'CMRE', 'CMS', 'CMSC', 'CMU', 'CNA', 'CNC', 'CNF', 'CNHI', 'CNI', 'CNK', 'CNMD', 'CNNE', 'CNO', 'CNP', 'CNQ', 'CNR', 'CNS', 'CNX', 'CNXM', 'CO', 'CODI', 'COE', 'COF', 'COG', 'COLD', 'COO', 'COP', 'COR', 'CORR', 'COTY', 'CP', 'CPA', 'CPAC', 'CPB', 'CPE', 'CPF', 'CPG', 'CPK', 'CPLG', 'CPRI', 'CPS', 'CPT', 'CR', 'CRC', 'CRH', 'CRI', 'CRK', 'CRL', 'CRM', 'CRS', 'CRT', 'CRY', 'CS', 'CSL', 'CSLT', 'CSPR', 'CSTM', 'CSU', 'CSV', 'CTAA', 'CTB', 'CTBB', 'CTK', 'CTL', 'CTLT', 'CTR', 'CTRA', 'CTS', 'CTT', 'CTV', 'CTVA', 'CTY', 'CUB', 'CUBE', 'CUBI', 'CUK', 'CULP', 'CURO', 'CUZ', 'CVA', 'CVE', 'CVEO', 'CVI', 'CVIA', 'CVNA', 'CVS', 'CVX', 'CW', 'CWEN', 'CWH', 'CWK', 'CWT', 'CX', 'CXE', 'CXH', 'CXO', 'CXP', 'CXW', 'CYD', 'CYH', 'CZZ', 'D', 'DAC', 'DAL', 'DAN', 'DAO', 'DAR', 'DAVA', 'DB', 'DBD', 'DBI', 'DBL', 'DCF', 'DCI', 'DCO', 'DCP', 'DCUE', 'DD', 'DDD', 'DDF', 'DDS', 'DDT', 'DE', 'DEA', 'DECK', 'DEI', 'DELL', 'DEO', 'DESP', 'DEX', 'DFIN', 'DFNS', 'DFP', 'DFS', 'DG', 'DGX', 'DHF', 'DHI', 'DHR', 'DHT', 'DHX', 'DIAX', 'DIN', 'DIS', 'DK', 'DKL', 'DKS', 'DL', 'DLB', 'DLNG', 'DLPH', 'DLR', 'DLX', 'DMB', 'DNK', 'DNOW', 'DNP', 'DNR', 'DO', 'DOC', 'DOOR', 'DOV', 'DOW', 'DPG', 'DPZ', 'DQ', 'DRD', 'DRE', 'DRH', 'DRI', 'DRQ', 'DRUA', 'DS', 'DSE', 'DSSI', 'DSX', 'DT', 'DTE', 'DTP', 'DTY', 'DUC', 'DUK', 'DUK-A', 'DUKB', 'DUKH', 'DVA', 'DVD', 'DVN', 'DX', 'DX-B', 'DX-C', 'DXB', 'DXC', 'DY', 'E', 'EAB', 'EAE', 'EAF', 'EAI', 'EARN', 'EAT', 'EB', 'EBF', 'EBR', 'EBR.B', 'EBS', 'EC', 'ECC', 'ECCB', 'ECCX', 'ECCY', 'ECL', 'ECOM', 'ECT', 'ED', 'EDD', 'EDF', 'EDI', 'EDN', 'EDU', 'EE', 'EEA', 'EEX', 'EFC', 'EFC-A', 'EFF', 'EFL', 'EFR', 'EFT', 'EFX', 'EGF', 'EGHT', 'EGIF', 'EGO', 'EGP', 'EGY', 'EHC', 'EHI', 'EHT', 'EIC', 'EIG', 'EIX', 'EL', 'ELAN', 'ELAT', 'ELC', 'ELF', 'ELJ', 'ELP', 'ELS', 'ELU', 'ELVT', 'ELY', 'EMD', 'EME', 'EMF', 'EMN', 'EMO', 'EMP', 'EMR', 'ENB', 'ENBA', 'ENBL', 'ENIA', 'ENIC', 'ENJ', 'ENLC', 'ENO', 'ENR', 'ENR-A', 'ENS', 'ENV', 'ENVA', 'ENZ', 'EOD', 'EOG', 'EOI', 'EOS', 'EOT', 'EP-C', 'EPAC', 'EPAM', 'EPC', 'EPD', 'EPR', 'EPR-C', 'EPR-E', 'EPR-G', 'EPRT', 'EQC', 'EQC-D', 'EQH', 'EQH-A', 'EQM', 'EQNR', 'EQR', 'EQS', 'EQT', 'ERA', 'ERF', 'ERJ', 'EROS', 'ES', 'ESE', 'ESI', 'ESNT', 'ESRT', 'ESS', 'ESTC', 'ESTE', 'ET', 'ETB', 'ETG', 'ETH', 'ETI.P', 'ETJ', 'ETM', 'ETN', 'ETO', 'ETP-C', 'ETP-D', 'ETP-E', 'ETR', 'ETRN', 'ETV', 'ETW', 'ETX', 'ETY', 'EURN', 'EV', 'EVA', 'EVC', 'EVF', 'EVG', 'EVH', 'EVN', 'EVR', 'EVRG', 'EVRI', 'EVT', 'EVTC', 'EW', 'EXD', 'EXG', 'EXK', 'EXP', 'EXPR', 'EXR', 'EXTN', 'EZT', 'F', 'F-B', 'F-C', 'FAF', 'FAM', 'FBC', 'FBHS', 'FBK', 'FBM', 'FBP', 'FC', 'FCAU', 'FCF', 'FCN', 'FCPT', 'FCT', 'FCX', 'FDEU', 'FDP', 'FDS', 'FDX', 'FE', 'FEA.U', 'FEA.W', 'FEAC', 'FEDU', 'FEI', 'FENG', 'FEO', 'FET', 'FF', 'FFA', 'FFC', 'FFG', 'FG', 'FG.W', 'FGB', 'FHI', 'FHN', 'FHN-A', 'FI', 'FICO', 'FIF', 'FINS', 'FINV', 'FIS', 'FIT', 'FIV', 'FIX', 'FL', 'FLC', 'FLNG', 'FLO', 'FLOW', 'FLR', 'FLS', 'FLT', 'FLY', 'FMC', 'FMN', 'FMO', 'FMS', 'FMX', 'FMY', 'FN', 'FNB', 'FNB-E', 'FND', 'FNF', 'FNV', 'FOE', 'FOF', 'FOR', 'FPA.U', 'FPA.W', 'FPAC', 'FPF', 'FPH', 'FPI', 'FPI-B', 'FPL', 'FR', 'FRA', 'FRC', 'FRC-F', 'FRC-G', 'FRC-H', 'FRC-I', 'FRC-J', 'FRO', 'FRT', 'FRT-C', 'FSB', 'FSD', 'FSK', 'FSLY', 'FSM', 'FSS', 'FT', 'FTA-A', 'FTA-B', 'FTAI', 'FTCH', 'FTI', 'FTK', 'FTS', 'FTSI', 'FTV', 'FTV-A', 'FUL', 'FUN', 'FVA.U', 'FVRR', 'G', 'GAB', 'GAB-G', 'GAB-H', 'GAB-J', 'GAB-K', 'GAM', 'GAM-B', 'GATX', 'GBAB', 'GBL', 'GBX', 'GCAP', 'GCI', 'GCO', 'GCP', 'GCV', 'GD', 'GDDY', 'GDL', 'GDL-C', 'GDO', 'GDOT', 'GDV', 'GDV-A', 'GDV-G', 'GDV-H', 'GE', 'GEF', 'GEF.B', 'GEL', 'GEN', 'GEO', 'GER', 'GES', 'GF', 'GFF', 'GFI', 'GFL', 'GFLU', 'GFY', 'GGB', 'GGG', 'GGM', 'GGT', 'GGT-E', 'GGT-G', 'GGZ', 'GGZ-A', 'GHC', 'GHG', 'GHL', 'GHM', 'GHY', 'GIB', 'GIL', 'GIM', 'GIS', 'GIX', 'GIX.P', 'GIX.U', 'GIX.W', 'GJH', 'GJO', 'GJP', 'GJR', 'GJS', 'GJT', 'GKOS', 'GL', 'GL-C', 'GLE.U', 'GLE.W', 'GLEO', 'GLO-A', 'GLO-B', 'GLO-C', 'GLO-G', 'GLOB', 'GLOG', 'GLOP', 'GLP', 'GLP-A', 'GLT', 'GLW', 'GM', 'GME', 'GMED', 'GMR-A', 'GMRE', 'GMS', 'GMTA', 'GMZ', 'GNC', 'GNE', 'GNE-A', 'GNK', 'GNL', 'GNL-A', 'GNL-B', 'GNRC', 'GNT', 'GNT-A', 'GNW', 'GOF', 'GOL', 'GOLD', 'GOLF', 'GOOS', 'GPC', 'GPI', 'GPJA', 'GPK', 'GPM', 'GPMT', 'GPN', 'GPRK', 'GPS', 'GPX', 'GRA', 'GRA.U', 'GRA.W', 'GRAF', 'GRAM', 'GRC', 'GRP.U', 'GRUB', 'GRX', 'GRX-B', 'GS', 'GS-A', 'GS-C', 'GS-D', 'GS-J', 'GS-K', 'GS-N', 'GSBD', 'GSH', 'GSK', 'GSL', 'GSL-B', 'GSLD', 'GSX', 'GTES', 'GTN', 'GTN.A', 'GTS', 'GTT', 'GTX', 'GTY', 'GUT', 'GUT-A', 'GUT-C', 'GVA', 'GWB', 'GWRE', 'GWW', 'GYC', 'H', 'HAE', 'HAL', 'HASI', 'HBB', 'HBI', 'HBM', 'HCA', 'HCC', 'HCFT', 'HCHC', 'HCI', 'HCR', 'HCXY', 'HCXZ', 'HD', 'HDB', 'HE', 'HEI', 'HEI.A', 'HEP', 'HEQ', 'HES', 'HESM', 'HEXO', 'HFC', 'HFR-A', 'HFRO', 'HGH', 'HGLB', 'HGV', 'HHC', 'HHS', 'HI', 'HIE', 'HIG', 'HIG-G', 'HII', 'HIL', 'HIO', 'HIW', 'HIX', 'HJV', 'HKIB', 'HL', 'HL-B', 'HLF', 'HLI', 'HLT', 'HLX', 'HMC', 'HMI', 'HML-A', 'HMLP', 'HMN', 'HMY', 'HNGR', 'HNI', 'HNP', 'HOG', 'HOME', 'HON', 'HOV', 'HP', 'HPE', 'HPF', 'HPI', 'HPP', 'HPQ', 'HPR', 'HPS', 'HQH', 'HQL', 'HR', 'HRB', 'HRC', 'HRI', 'HRL', 'HRTG', 'HSB-A', 'HSBC', 'HSC', 'HST', 'HSY', 'HT', 'HT-C', 'HT-D', 'HT-E', 'HTA', 'HTD', 'HTFA', 'HTGC', 'HTH', 'HTY', 'HTZ', 'HUBB', 'HUBS', 'HUD', 'HUM', 'HUN', 'HUYA', 'HVT', 'HVT.A', 'HWM', 'HXL', 'HY', 'HYB', 'HYI', 'HYT', 'HZN', 'HZO', 'I', 'IAA', 'IAE', 'IAG', 'IBA', 'IBM', 'IBN', 'IBP', 'ICD', 'ICE', 'ICL', 'IDA', 'IDE', 'IDT', 'IEX', 'IFF', 'IFFT', 'IFN', 'IFS', 'IGA', 'IGD', 'IGI', 'IGR', 'IGT', 'IHC', 'IHD', 'IHG', 'IHIT', 'IHTA', 'IID', 'IIF', 'IIM', 'IIP-A', 'IIPR', 'IMAX', 'INFO', 'INFY', 'ING', 'INGR', 'INN', 'INN-D', 'INN-E', 'INS-A', 'INSI', 'INSP', 'INSW', 'INT', 'INVH', 'IO', 'IP', 'IPG', 'IPHI', 'IPI', 'IPO.U', 'IPV', 'IPV.U', 'IPV.W', 'IQI', 'IQV', 'IR', 'IRE-C', 'IRET', 'IRL', 'IRM', 'IRR', 'IRS', 'IRT', 'ISD', 'ISG', 'IT', 'ITCB', 'ITGR', 'ITT', 'ITUB', 'ITW', 'IVC', 'IVH', 'IVR', 'IVR-A', 'IVR-B', 'IVR-C', 'IVZ', 'IX', 'J', 'JAX', 'JBGS', 'JBK', 'JBL', 'JBN', 'JBR', 'JBT', 'JCA-B', 'JCAP', 'JCE', 'JCI', 'JCO', 'JCP', 'JDD', 'JE', 'JE-A', 'JEF', 'JELD', 'JEMD', 'JEQ', 'JFR', 'JGH', 'JHAA', 'JHB', 'JHG', 'JHI', 'JHS', 'JHX', 'JHY', 'JIH', 'JIH.U', 'JIH.W', 'JILL', 'JKS', 'JLL', 'JLS', 'JMEI', 'JMF', 'JMIA', 'JMLP', 'JMM', 'JMP', 'JNJ', 'JNPR', 'JOE', 'JOF', 'JP', 'JPC', 'JPI', 'JPM', 'JPM-C', 'JPM-D', 'JPM-G', 'JPM-H', 'JPM-J', 'JPS', 'JPT', 'JQC', 'JRI', 'JRO', 'JRS', 'JSD', 'JT', 'JTA', 'JTD', 'JW.A', 'JW.B', 'JWN', 'K', 'KAI', 'KAMN', 'KAR', 'KB', 'KBH', 'KBR', 'KDMN', 'KDP', 'KEM', 'KEN', 'KEP', 'KEX', 'KEY', 'KEY-I', 'KEY-J', 'KEY-K', 'KEYS', 'KF', 'KFS', 'KFY', 'KGC', 'KIM', 'KIM-L', 'KIM-M', 'KIO', 'KKR', 'KKR-A', 'KKR-B', 'KL', 'KMB', 'KMF', 'KMI', 'KMPR', 'KMT', 'KMX', 'KN', 'KNL', 'KNOP', 'KNX', 'KO', 'KODK', 'KOF', 'KOP', 'KOS', 'KR', 'KRA', 'KRC', 'KREF', 'KRG', 'KRO', 'KRP', 'KSM', 'KSS', 'KSU', 'KSU.P', 'KT', 'KTB', 'KTF', 'KTH', 'KTN', 'KTP', 'KW', 'KWR', 'KYN', 'L', 'LAC', 'LAD', 'LADR', 'LAIX', 'LAZ', 'LB', 'LBRT', 'LC', 'LCI', 'LCII', 'LDL', 'LDOS', 'LDP', 'LEA', 'LEAF', 'LEE', 'LEG', 'LEJU', 'LEN', 'LEN.B', 'LEO', 'LEVI', 'LFC', 'LGC', 'LGC.U', 'LGC.W', 'LGF.A', 'LGF.B', 'LGI', 'LH', 'LHC', 'LHC.U', 'LHC.W', 'LHX', 'LII', 'LIN', 'LINX', 'LITB', 'LL', 'LLY', 'LM', 'LMHA', 'LMHB', 'LMT', 'LN', 'LNC', 'LND', 'LNN', 'LOMA', 'LOW', 'LPG', 'LPI', 'LPL', 'LPX', 'LRN', 'LSI', 'LTC', 'LTHM', 'LTM', 'LUB', 'LUV', 'LVS', 'LW', 'LXFR', 'LXP', 'LXP-C', 'LXU', 'LYB', 'LYG', 'LYV', 'LZB', 'M', 'MA', 'MAA', 'MAA-I', 'MAC', 'MAIN', 'MAN', 'MANU', 'MAS', 'MATX', 'MAV', 'MAXR', 'MBI', 'MBT', 'MC', 'MCA', 'MCB', 'MCC', 'MCD', 'MCI', 'MCK', 'MCN', 'MCO', 'MCR', 'MCS', 'MCV', 'MCX', 'MCY', 'MD', 'MDC', 'MDLA', 'MDLQ', 'MDLX', 'MDLY', 'MDP', 'MDT', 'MDU', 'MEC', 'MED', 'MEI', 'MEN', 'MER-K', 'MET', 'MET-A', 'MET-E', 'MET-F', 'MFA', 'MFA-B', 'MFA-C', 'MFA.U', 'MFA.W', 'MFAC', 'MFC', 'MFD', 'MFG', 'MFGP', 'MFL', 'MFM', 'MFO', 'MFT', 'MFV', 'MG', 'MGA', 'MGF', 'MGM', 'MGP', 'MGR', 'MGU', 'MGY', 'MH-A', 'MH-C', 'MH-D', 'MHD', 'MHE', 'MHF', 'MHI', 'MHK', 'MHLA', 'MHN', 'MHNC', 'MHO', 'MIC', 'MIE', 'MIN', 'MIT-A', 'MIT-B', 'MIT-C', 'MITT', 'MIXT', 'MIY', 'MKC', 'MKC.V', 'MKL', 'MLI', 'MLM', 'MLP', 'MLR', 'MMC', 'MMD', 'MMI', 'MMM', 'MMP', 'MMS', 'MMT', 'MMU', 'MN', 'MNE', 'MNK', 'MNP', 'MNR', 'MNR-C', 'MNRL', 'MO', 'MOD', 'MODN', 'MOG.A', 'MOG.B', 'MOGU', 'MOH', 'MOS', 'MOV', 'MPA', 'MPC', 'MPLX', 'MPV', 'MPW', 'MPX', 'MQT', 'MQY', 'MR', 'MRC', 'MRK', 'MRO', 'MS', 'MS-A', 'MS-E', 'MS-F', 'MS-I', 'MS-K', 'MS-L', 'MSA', 'MSB', 'MSC', 'MSCI', 'MSD', 'MSG', 'MSGE', 'MSGN', 'MSGS', 'MSI', 'MSM', 'MT', 'MTB', 'MTD', 'MTDR', 'MTG', 'MTH', 'MTL', 'MTL.P', 'MTN', 'MTOR', 'MTR', 'MTRN', 'MTT', 'MTW', 'MTX', 'MTZ', 'MUA', 'MUC', 'MUE', 'MUFG', 'MUH', 'MUI', 'MUJ', 'MUR', 'MUS', 'MUSA', 'MUX', 'MVC', 'MVCD', 'MVF', 'MVO', 'MVT', 'MWA', 'MX', 'MXE', 'MXF', 'MXL', 'MYC', 'MYD', 'MYE', 'MYF', 'MYI', 'MYJ', 'MYN', 'MYOV', 'MZA', 'NAC', 'NAD', 'NAN', 'NAT', 'NAV', 'NAV-D', 'NAZ', 'NBB', 'NBHC', 'NBR', 'NBR-A', 'NC', 'NCA', 'NCB', 'NCLH', 'NCR', 'NCV', 'NCV-A', 'NCZ', 'NCZ-A', 'NDP', 'NE', 'NEA', 'NEE', 'NEE-I', 'NEE-J', 'NEE-K', 'NEE-N', 'NEE-O', 'NEE-P', 'NEM', 'NEP', 'NET', 'NEU', 'NEV', 'NEW', 'NEWR', 'NEX', 'NEXA', 'NFG', 'NFH', 'NFH.W', 'NFJ', 'NGG', 'NGL', 'NGL-A', 'NGL-B', 'NGL-C', 'NGS', 'NGVC', 'NGVT', 'NHA', 'NHF', 'NHI', 'NI', 'NI-B', 'NID', 'NIE', 'NIM', 'NINE', 'NIO', 'NIQ', 'NJR', 'NJV', 'NKE', 'NKG', 'NKX', 'NL', 'NLS', 'NLSN', 'NLY', 'NLY-D', 'NLY-F', 'NLY-G', 'NLY-I', 'NM', 'NM-G', 'NM-H', 'NMCO', 'NMFC', 'NMFX', 'NMI', 'NMK-B', 'NMK-C', 'NMM', 'NMR', 'NMS', 'NMT', 'NMY', 'NMZ', 'NNA', 'NNI', 'NNN', 'NNN-F', 'NNY', 'NOA', 'NOAH', 'NOC', 'NOK', 'NOM', 'NOMD', 'NOV', 'NOVA', 'NOW', 'NP', 'NPK', 'NPN', 'NPO', 'NPTN', 'NPV', 'NQP', 'NR', 'NREF', 'NRG', 'NRGX', 'NRK', 'NRP', 'NRT', 'NRUC', 'NRZ', 'NRZ-A', 'NRZ-B', 'NRZ-C', 'NS', 'NS-A', 'NS-B', 'NS-C', 'NSA', 'NSA-A', 'NSC', 'NSC.W', 'NSCO', 'NSL', 'NSP', 'NSS', 'NTB', 'NTCO', 'NTEST.I', 'NTEST.J', 'NTEST.K', 'NTG', 'NTP', 'NTR', 'NTZ', 'NUE', 'NUM', 'NUO', 'NUS', 'NUV', 'NUW', 'NVG', 'NVGS', 'NVO', 'NVR', 'NVRO', 'NVS', 'NVST', 'NVT', 'NVTA', 'NWE', 'NWHM', 'NWN', 'NX', 'NXC', 'NXJ', 'NXN', 'NXP', 'NXQ', 'NXR', 'NXRT', 'NYC-A', 'NYC-U', 'NYCB', 'NYT', 'NYV', 'NZF', 'O', 'OAC', 'OAC.U', 'OAC.W', 'OAK-A', 'OAK-B', 'OC', 'OCFT', 'OCN', 'ODC', 'OEC', 'OFC', 'OFG', 'OFG-A', 'OFG-B', 'OFG-D', 'OGE', 'OGS', 'OHI', 'OI', 'OIA', 'OIB.C', 'OII', 'OIS', 'OKE', 'OLN', 'OLP', 'OMC', 'OMF', 'OMI', 'ONDK', 'ONE', 'ONTO', 'OOMA', 'OPP', 'OPY', 'OR', 'ORA', 'ORAN', 'ORC', 'ORCC', 'ORCL', 'ORI', 'ORN', 'OSB', 'OSG', 'OSK', 'OTIS', 'OUT', 'OVV', 'OXM', 'OXY', 'PAA', 'PAC', 'PAC.W', 'PACD', 'PACK', 'PAG', 'PAGP', 'PAGS', 'PAI', 'PAM', 'PANW', 'PAR', 'PARR', 'PAYC', 'PB', 'PBA', 'PBB', 'PBC', 'PBF', 'PBFX', 'PBH', 'PBI', 'PBI-B', 'PBR', 'PBR.A', 'PBT', 'PBY', 'PCF', 'PCG', 'PCI', 'PCK', 'PCM', 'PCN', 'PCP.U', 'PCQ', 'PD', 'PDI', 'PDM', 'PDS', 'PDT', 'PE', 'PEAK', 'PEB', 'PEB-C', 'PEB-D', 'PEB-E', 'PEB-F', 'PEG', 'PEI', 'PEI-B', 'PEI-C', 'PEI-D', 'PEN', 'PEO', 'PER', 'PFD', 'PFE', 'PFGC', 'PFH', 'PFL', 'PFN', 'PFO', 'PFS', 'PFSI', 'PG', 'PGP', 'PGR', 'PGRE', 'PGTI', 'PGZ', 'PH', 'PHD', 'PHG', 'PHI', 'PHK', 'PHM', 'PHR', 'PHT', 'PHX', 'PIC', 'PIC.U', 'PIC.W', 'PII', 'PIM', 'PINE', 'PING', 'PINS', 'PIPR', 'PIY', 'PJH', 'PJT', 'PK', 'PKE', 'PKG', 'PKI', 'PKO', 'PKX', 'PLAN', 'PLD', 'PLNT', 'PLOW', 'PLT', 'PLYM', 'PM', 'PMF', 'PML', 'PMM', 'PMO', 'PMT', 'PMT-A', 'PMT-B', 'PMX', 'PNC', 'PNC-P', 'PNC-Q', 'PNF', 'PNI', 'PNM', 'PNR', 'PNW', 'POL', 'POR', 'POST', 'PPG', 'PPL', 'PPR', 'PPT', 'PPX', 'PQG', 'PRA', 'PRE-F', 'PRE-G', 'PRE-H', 'PRE-I', 'PRGO', 'PRH', 'PRI', 'PRI-A', 'PRI-B', 'PRI-C', 'PRI-D', 'PRI-E', 'PRI-F', 'PRLB', 'PRMW', 'PRO', 'PROS', 'PRS', 'PRSP', 'PRT', 'PRTY', 'PRU', 'PSA', 'PSA-B', 'PSA-C', 'PSA-D', 'PSA-E', 'PSA-F', 'PSA-G', 'PSA-H', 'PSA-I', 'PSA-J', 'PSA-K', 'PSA-V', 'PSA-W', 'PSA-X', 'PSB', 'PSB-W', 'PSB-X', 'PSB-Y', 'PSB-Z', 'PSF', 'PSN', 'PSO', 'PSTG', 'PSTL', 'PSV', 'PSX', 'PSXP', 'PTR', 'PTY', 'PUK', 'PUK-A', 'PUK.P', 'PUMP', 'PVG', 'PVH', 'PVL', 'PWR', 'PXD', 'PYN', 'PYS', 'PYT', 'PYX', 'PZC', 'PZN', 'QD', 'QEP', 'QES', 'QGEN', 'QSR', 'QTS', 'QTS-A', 'QTS-B', 'QTWO', 'QUAD', 'QUOT', 'QVCC', 'QVCD', 'R', 'RA', 'RACE', 'RAD', 'RAMP', 'RBA', 'RBC', 'RBS', 'RC', 'RCA', 'RCB', 'RCI', 'RCL', 'RCP', 'RCS', 'RCUS', 'RDN', 'RDS.A', 'RDS.B', 'RDY', 'RE', 'RELX', 'RENN', 'RES', 'RESI', 'REV', 'REVG', 'REX', 'REX-A', 'REX-B', 'REX-C', 'REXR', 'REZI', 'RF', 'RF-A', 'RF-B', 'RF-C', 'RFI', 'RFL', 'RFM', 'RFP', 'RGA', 'RGR', 'RGS', 'RGT', 'RH', 'RHI', 'RHP', 'RIG', 'RIO', 'RIV', 'RJF', 'RL', 'RLGY', 'RLH', 'RLI', 'RLJ', 'RLJ-A', 'RM', 'RMAX', 'RMD', 'RMED', 'RMG', 'RMG.U', 'RMG.W', 'RMI', 'RMM', 'RMP.P', 'RMT', 'RNG', 'RNGR', 'RNP', 'RNR', 'RNR-E', 'RNR-F', 'ROG', 'ROK', 'ROL', 'ROP', 'ROYT', 'RPAI', 'RPL.U', 'RPL.W', 'RPLA', 'RPM', 'RPT', 'RPT-D', 'RQI', 'RRC', 'RRD', 'RRTS', 'RS', 'RSF', 'RSG', 'RST', 'RTW', 'RTX', 'RUBI', 'RVI', 'RVLV', 'RVT', 'RWT', 'RXN', 'RY', 'RY-T', 'RYAM', 'RYB', 'RYCE', 'RYI', 'RYN', 'RZA', 'RZB', 'SA', 'SAF', 'SAFE', 'SAH', 'SAIC', 'SAIL', 'SALT', 'SAM', 'SAN', 'SAN-B', 'SAND', 'SAP', 'SAR', 'SAVE', 'SB', 'SB-C', 'SB-D', 'SBE', 'SBE.U', 'SBE.W', 'SBH', 'SBI', 'SBNA', 'SBOW', 'SBR', 'SBS', 'SBSW', 'SC', 'SCA', 'SCCO', 'SCD', 'SCE-G', 'SCE-H', 'SCE-J', 'SCE-K', 'SCE-L', 'SCH-C', 'SCH-D', 'SCHW', 'SCI', 'SCL', 'SCM', 'SCP.U', 'SCP.W', 'SCPE', 'SCS', 'SCU', 'SCV.U', 'SCV.W', 'SCVX', 'SCX', 'SD', 'SDRL', 'SE', 'SEAS', 'SEE', 'SEM', 'SERV', 'SF', 'SF-A', 'SF-B', 'SFB', 'SFE', 'SFL', 'SFT.U', 'SFT.W', 'SFTW', 'SFUN', 'SGU', 'SHAK', 'SHG', 'SHI', 'SHL.W', 'SHLL', 'SHLX', 'SHO', 'SHO-E', 'SHO-F', 'SHOP', 'SHW', 'SI', 'SID', 'SIG', 'SIT-A', 'SIT-K', 'SITC', 'SITE', 'SIX', 'SJI', 'SJIJ', 'SJIU', 'SJM', 'SJR', 'SJT', 'SJW', 'SKM', 'SKT', 'SKX', 'SKY', 'SLB', 'SLCA', 'SLF', 'SLG', 'SLG-I', 'SM', 'SMAR', 'SMFG', 'SMG', 'SMHI', 'SMLP', 'SMM', 'SMP', 'SNA', 'SNAP', 'SNDR', 'SNE', 'SNN', 'SNP', 'SNR', 'SNV', 'SNV-D', 'SNV-E', 'SNX', 'SO', 'SOGO', 'SOI', 'SOJA', 'SOJB', 'SOJC', 'SOJD', 'SOL', 'SOLN', 'SON', 'SOR', 'SPA.U', 'SPA.W', 'SPAQ', 'SPB', 'SPCE', 'SPE', 'SPE-B', 'SPG', 'SPG-J', 'SPGI', 'SPH', 'SPL-A', 'SPLP', 'SPN', 'SPOT', 'SPR', 'SPXC', 'SPXX', 'SQ', 'SQM', 'SQNS', 'SR', 'SR-A', 'SRC', 'SRC-A', 'SRE', 'SRE-A', 'SRE-B', 'SREA', 'SRF', 'SRG', 'SRG-A', 'SRI', 'SRL', 'SRLP', 'SRT', 'SRV', 'SSD', 'SSI', 'SSL', 'SSTK', 'ST', 'STA-C', 'STA-D', 'STA-G', 'STA-I', 'STAG', 'STAR', 'STC', 'STE', 'STG', 'STK', 'STL', 'STL-A', 'STM', 'STN', 'STNG', 'STON', 'STOR', 'STT', 'STT-D', 'STT-G', 'STWD', 'STZ', 'STZ.B', 'SU', 'SUI', 'SUM', 'SUN', 'SUP', 'SUPV', 'SUZ', 'SWCH', 'SWI', 'SWK', 'SWM', 'SWN', 'SWP', 'SWT', 'SWX', 'SWZ', 'SXC', 'SXI', 'SXT', 'SYF', 'SYF-A', 'SYK', 'SYX', 'SYY', 'SZC', 'T', 'T-A', 'T-C', 'TAC', 'TAK', 'TAL', 'TALO', 'TAP', 'TAP.A', 'TARO', 'TBB', 'TBC', 'TBI', 'TCI', 'TCO', 'TCO-J', 'TCO-K', 'TCP', 'TCRW', 'TCRZ', 'TCS', 'TD', 'TDA', 'TDC', 'TDE', 'TDF', 'TDG', 'TDI', 'TDJ', 'TDOC', 'TDS', 'TDW', 'TDW.A', 'TDW.B', 'TDY', 'TEAF', 'TECK', 'TEF', 'TEI', 'TEL', 'TEN', 'TEO', 'TEVA', 'TEX', 'TFC', 'TFC-F', 'TFC-G', 'TFC-H', 'TFC-I', 'TFII', 'TFX', 'TG', 'TGE', 'TGH', 'TGI', 'TGNA', 'TGP', 'TGP-A', 'TGP-B', 'TGS', 'TGT', 'THC', 'THG', 'THGA', 'THO', 'THQ', 'THR', 'THS', 'THW', 'TIF', 'TISI', 'TJX', 'TK', 'TKC', 'TKR', 'TLI', 'TLK', 'TLRD', 'TLYS', 'TM', 'TME', 'TMHC', 'TMO', 'TMST', 'TNC', 'TNET', 'TNK', 'TNP', 'TNP-C', 'TNP-D', 'TNP-E', 'TNP-F', 'TOL', 'TOT', 'TPB', 'TPC', 'TPH', 'TPL', 'TPR', 'TPRE', 'TPVG', 'TPVY', 'TPX', 'TPZ', 'TR', 'TRC', 'TREC', 'TREX', 'TRGP', 'TRI', 'TRN', 'TRN.U', 'TRN.W', 'TRNE', 'TRNO', 'TROX', 'TRP', 'TRQ', 'TRT-A', 'TRT-B', 'TRT-C', 'TRT-D', 'TRTN', 'TRTX', 'TRU', 'TRV', 'TRWH', 'TS', 'TSE', 'TSI', 'TSLF', 'TSLX', 'TSM', 'TSN', 'TSQ', 'TSU', 'TT', 'TTC', 'TTI', 'TTM', 'TTP', 'TU', 'TUFN', 'TUP', 'TV', 'TVC', 'TVE', 'TWI', 'TWLO', 'TWN', 'TWO', 'TWO-A', 'TWO-B', 'TWO-C', 'TWO-D', 'TWO-E', 'TWTR', 'TX', 'TXT', 'TY', 'TY.P', 'TYG', 'TYL', 'UA', 'UAA', 'UAN', 'UBA', 'UBER', 'UBP', 'UBP-H', 'UBP-K', 'UBS', 'UDR', 'UE', 'UFI', 'UFS', 'UGI', 'UGP', 'UHS', 'UHT', 'UI', 'UIS', 'UL', 'UMC', 'UMH', 'UMH-B', 'UMH-C', 'UMH-D', 'UN', 'UNF', 'UNFI', 'UNH', 'UNM', 'UNMA', 'UNP', 'UNT', 'UNVR', 'UPS', 'URI', 'USA', 'USAC', 'USB', 'USB-A', 'USB-H', 'USB-M', 'USB-O', 'USB-P', 'USDP', 'USFD', 'USM', 'USNA', 'USPH', 'USX', 'UTF', 'UTI', 'UTL', 'UVE', 'UVV', 'UZA', 'UZB', 'UZC', 'V', 'VAC', 'VAL', 'VALE', 'VAM', 'VAPO', 'VAR', 'VBF', 'VCIF', 'VCRA', 'VCV', 'VEC', 'VEDL', 'VEEV', 'VEL', 'VER', 'VER-F', 'VER.U', 'VET', 'VFC', 'VGI', 'VGM', 'VGR', 'VHI', 'VICI', 'VIPS', 'VIST', 'VIV', 'VJET', 'VKQ', 'VLO', 'VLRS', 'VLT', 'VMC', 'VMI', 'VMO', 'VMW', 'VNCE', 'VNE', 'VNO', 'VNO-K', 'VNO-L', 'VNO-M', 'VNTR', 'VOC', 'VOY-B', 'VOYA', 'VPG', 'VPV', 'VRS', 'VRT', 'VRT.W', 'VRTV', 'VSH', 'VSLR', 'VST', 'VST.A', 'VSTO', 'VTA', 'VTN', 'VTR', 'VVI', 'VVN.W', 'VVNT', 'VVR', 'VVV', 'VZ', 'W', 'WAB', 'WAL', 'WALA', 'WAT', 'WBAI', 'WBC', 'WBK', 'WBS', 'WBS-F', 'WBT', 'WCC', 'WCN', 'WD', 'WDR', 'WEA', 'WEC', 'WEI', 'WELL', 'WES', 'WEX', 'WF', 'WFC', 'WFC-L', 'WFC-N', 'WFC-O', 'WFC-P', 'WFC-Q', 'WFC-R', 'WFC-T', 'WFC-V', 'WFC-W', 'WFC-X', 'WFC-Y', 'WFC-Z', 'WGO', 'WH', 'WHD', 'WHG', 'WHR', 'WIA', 'WIT', 'WIW', 'WK', 'WLK', 'WLKP', 'WLL', 'WM', 'WMB', 'WMC', 'WMK', 'WMS', 'WMT', 'WNC', 'WNS', 'WOR', 'WORK', 'WOW', 'WPC', 'WPG', 'WPG-H', 'WPG-I', 'WPM', 'WPP', 'WPX', 'WRB', 'WRB-B', 'WRB-C', 'WRB-D', 'WRB-E', 'WRB-F', 'WRE', 'WRI', 'WRK', 'WSM', 'WSO', 'WSO.B', 'WSR', 'WST', 'WTI', 'WTM', 'WTRG', 'WTRU', 'WTS', 'WTTR', 'WU', 'WUBA', 'WWE', 'WWW', 'WY', 'WYND', 'X', 'XAN', 'XAN-C', 'XEC', 'XFLT', 'XHR', 'XIN', 'XOM', 'XPO', 'XRF', 'XRX', 'XYF', 'XYL', 'Y', 'YELP', 'YETI', 'YEXT', 'YPF', 'YRD', 'YUM', 'YUMC', 'ZBH', 'ZEN', 'ZNH', 'ZTO', 'ZTR', 'ZTS', 'ZUO', 'ZYME']



trashTicks = ['BSMX', 'OTIS', 'OUT', 'OVV', 'OXY', 'PAA', 'PAC.W', 'PACD', 'PACK', 'PAGP', 'PAI', 'PANW', 'PAR', 'PARR', 'PBB', 'PBC', 'PBI', 'PBI-B', 'PBR', 'PBR.A', 'PBY', 'PCG', 'PCI', 'PCP.U', 'PD', 'PDI', 'PDS', 'PE', 'PEAK', 'PEB-C', 'PEB-D', 'PEB-E', 'PEB-F', 'PEG', 'PEI', 'PEI-B', 'PEI-C', 'PEI-D', 'PFH', 'PGRE', 'PGZ', 'PHG', 'PHK', 'PHM', 'PHR', 'PHX', 'PIC', 'PIC.U', 'PIC.W', 'PINE', 'PING', 'PINS', 'PIPR', 'PIY', 'PJH', 'PKI', 'PKO', 'PLAN', 'PLT', 'PLYM', 'PMT-A', 'PMT-B', 'PNC-P', 'PNC-Q', 'PNR', 'PNW', 'POL', 'POST', 'PPL', 'PPR', 'PPX', 'PRA', 'PRE-F', 'PRE-G', 'PRE-H', 'PRE-I', 'PRH', 'PRI-A', 'PRI-B', 'PRI-C', 'PRI-D', 'PRI-E', 'PRI-F', 'PRMW', 'PRO', 'PRS', 'PRTY', 'PSA-B', 'PSA-C', 'PSA-D', 'PSA-E', 'PSA-F', 'PSA-G', 'PSA-H', 'PSA-I', 'PSA-J', 'PSA-K', 'PSA-V', 'PSA-W', 'PSA-X', 'PSB-W', 'PSB-X', 'PSB-Y', 'PSB-Z', 'PSF', 'PSTG', 'PSTL', 'PSV', 'PUK-A', 'PUK.P', 'PVL', 'PXD', 'PYS', 'PYT', 'PYX', 'QD', 'QES', 'QGEN', 'QTS', 'QTS-A', 'QTS-B', 'QTWO', 'QUAD', 'QUOT', 'QVCC', 'QVCD', 'R', 'RA', 'RACE', 'RAD', 'RAMP', 'RCA', 'RCB', 'RCI', 'RCP', 'RCUS', 'RDN', 'RDS.A', 'RDS.B', 'RE', 'RELX', 'RENN', 'RES', 'RESI', 'REV', 'REVG', 'REX', 'REX-A', 'REX-B', 'REX-C', 'REZI', 'RF-A', 'RF-B', 'RF-C', 'RFL', 'RFM', 'RFP', 'RGA', 'RGS', 'RGT', 'RHI', 'RHP', 'RIG', 'RIO', 'RIV', 'RJF', 'RL', 'RLGY', 'RLH', 'RLJ-A', 'RMED', 'RMG.U', 'RMG.W', 'RMI', 'RMM', 'RMP.P', 'RNG', 'RNR-E', 'RNR-F', 'RPL.U', 'RPL.W', 'RPT', 'RPT-D', 'RRC', 'RRD', 'RRTS', 'RSF', 'RST', 'RTW', 'RTX', 'RUBI', 'RVLV', 'RWT', 'RY-T', 'RYAM', 'RYB', 'RYCE', 'RZA', 'RZB', 'SA', 'SAF', 'SAH', 'SAIL', 'SALT', 'SAN', 'SAN-B', 'SAND', 'SAR', 'SB-C', 'SB-D', 'SBE', 'SBE.U', 'SBE.W', 'SBI', 'SBNA', 'SBSW', 'SCA', 'SCE-G', 'SCE-H', 'SCE-J', 'SCE-K', 'SCE-L', 'SCH-C', 'SCH-D', 'SCP.U', 'SCP.W', 'SCPE', 'SCU', 'SCV.U', 'SCV.W', 'SCVX', 'SD', 'SDRL', 'SE', 'SEE', 'SF-A', 'SF-B', 'SFB', 'SFE', 'SFT.U', 'SFT.W', 'SFTW', 'SFUN', 'SHI', 'SHL.W', 'SHO', 'SHO-E', 'SHO-F', 'SHOP', 'SIG', 'SIT-A', 'SIT-K', 'SITC', 'SJIJ', 'SJIU', 'SJM', 'SKM', 'SLB', 'SLCA', 'SLG-I', 'SM', 'SMAR', 'SMFG', 'SMHI', 'SMLP', 'SMM', 'SNAP', 'SNDR', 'SNN', 'SNV', 'SNV-D', 'SNV-E', 'SOJA', 'SOJB', 'SOJC', 'SOJD', 'SOL', 'SOLN', 'SPA.U', 'SPA.W', 'SPB', 'SPCE', 'SPE', 'SPE-B', 'SPG', 'SPG-J', 'SPGI', 'SPL-A', 'SPLP', 'SPN', 'SPOT', 'SQNS', 'SR', 'SR-A', 'SRC', 'SRC-A', 'SRE-A', 'SRE-B', 'SREA', 'SRF', 'SRG', 'SRG-A', 'SRL', 'SRT', 'SRV', 'SSI', 'SSL', 'STA-C', 'STA-D', 'STA-G', 'STA-I', 'STAR', 'STE', 'STG', 'STK', 'STL', 'STL-A', 'STM', 'STN', 'STNG', 'STON', 'STT', 'STT-D', 'STT-G', 'STZ', 'STZ.B', 'SUP', 'SUZ', 'SWN', 'SWP', 'SWT', 'SXC', 'SYF-A', 'SYX', 'SZC', 'T-A', 'T-C', 'TAC', 'TAK', 'TAL', 'TAP', 'TAP.A', 'TBB', 'TBC', 'TBI', 'TCI', 'TCO-J', 'TCO-K', 'TCRW', 'TCRZ', 'TDA', 'TDE', 'TDI', 'TDJ', 'TDOC', 'TDS', 'TDW', 'TDW.A', 'TDW.B', 'TEAF', 'TEF', 'TEI', 'TEL', 'TEN', 'TEO', 'TEVA', 'TEX', 'TFC', 'TFC-F', 'TFC-G', 'TFC-H', 'TFC-I', 'TFII', 'TGE', 'TGI', 'TGNA', 'TGP-A', 'TGP-B', 'TGT', 'THC', 'THG', 'THGA', 'THQ', 'THS', 'THW', 'TISI', 'TK', 'TKC', 'TLI', 'TLRD', 'TME', 'TMHC', 'TMST', 'TNK', 'TNP', 'TNP-C', 'TNP-D', 'TNP-E', 'TNP-F', 'TPC', 'TPR', 'TPRE', 'TPVY', 'TPZ', 'TREC', 'TRGP', 'TRI', 'TRN.U', 'TRN.W', 'TRNE', 'TROX', 'TRQ', 'TRT-A', 'TRT-B', 'TRT-C', 'TRT-D', 'TRV', 'TS', 'TSLF', 'TSQ', 'TSU', 'TT', 'TTI', 'TTP', 'TU', 'TUFN', 'TUP', 'TVC', 'TVE', 'TWI', 'TWLO', 'TWO', 'TWO-A', 'TWO-B', 'TWO-C', 'TWO-D', 'TWO-E', 'TY.P', 'TYG', 'TYL', 'UAN', 'UBER', 'UBP', 'UBP-H', 'UBP-K', 'UFI', 'UFS', 'UI', 'UIS', 'UL', 'UMC', 'UMH', 'UMH-B', 'UMH-C', 'UMH-D', 'UN', 'UNF', 'UNFI', 'UNH', 'UNMA', 'UNT', 'UNVR', 'USA', 'USAC', 'USB', 'USB-A', 'USB-H', 'USB-M', 'USB-O', 'USB-P', 'USDP', 'USX', 'UVE', 'UZA', 'UZB', 'UZC', 'VAL', 'VALE', 'VAM', 'VAPO', 'VAR', 'VCIF', 'VCRA', 'VEDL', 'VER', 'VER-F', 'VER.U', 'VET', 'VGI', 'VGR', 'VIST', 'VIV', 'VJET', 'VLO', 'VLRS', 'VNCE', 'VNE', 'VNO', 'VNO-K', 'VNO-L', 'VNO-M', 'VNTR', 'VOY-B', 'VOYA', 'VRT', 'VRT.W', 'VRTV', 'VSLR', 'VST', 'VST.A', 'VSTO', 'VTA', 'VTR', 'VVI', 'VVN.W', 'VVNT', 'VZ', 'W', 'WALA', 'WBAI', 'WBS-F', 'WBT', 'WELL', 'WF', 'WFC', 'WFC-L', 'WFC-N', 'WFC-O', 'WFC-P', 'WFC-Q', 'WFC-R', 'WFC-T', 'WFC-V', 'WFC-W', 'WFC-X', 'WFC-Y', 'WFC-Z', 'WK', 'WLL', 'WM', 'WMC', 'WMS', 'WORK', 'WPG', 'WPG-H', 'WPG-I', 'WPM', 'WPP', 'WPX', 'WRB-B', 'WRB-C', 'WRB-D', 'WRB-E', 'WRB-F', 'WRI', 'WSO.B', 'WST', 'WTM', 'WTRG', 'WTRU', 'WTTR', 'X', 'XAN', 'XAN-C', 'XEC', 'XFLT', 'XOM', 'XPO', 'XRF', 'YEXT', 'YUM', 'ZEN', 'ZTO', 'ZUO', 'ZYME']
for tick in trashTicks:
    tickers.remove(tick)

#tickers = shuffle(tickers)

# import only system from os 
from os import system, name 

# import sleep to show output for some time period 
from time import sleep 

# define our clear function 
def clear():
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear')

def series(ticker):
    #print(ticker)
    data                =   si.get_data(ticker)
    if(len(data.iloc[:,4])<=253*5):
      return 0,0  
    #print("prediction:"+str(predictions[-1][0])+", price:"+str(price[-1][0])+" ratio"+str(())
    return (price[-1][0]/price[-30][0])-1

table = pd.DataFrame(data={
    'ticker':['0'],
    'growth':[0],
    'price':[0],
    'trailingAnnualDividendYield':[0],
    'payoutRatio':[0],
    'trailingAnnualDividendRate':[0],
    'dividendRate':[0],
    'priceToSalesTrailing12Months':[0],
    'forwardPE':[0],
    'fiveYearAvgDividendYield':[0],
    'dividendYield':[0],
    'fiveYearAvgDividendYield':[0],
    'profitMargins':[0],
    'forwardEps':[0],
    'bookValue':[0],
    'trailingEps':[0],
    'priceToBook':[0],
    'pegRatio':[0]})

array = ['ticker',
        'growth',
        'trailingAnnualDividendYield',
        'payoutRatio',
        'trailingAnnualDividendRate',
        'dividendRate',
        'trailingPE',
        'strikePrice',
        'priceToSalesTrailing12Months',
        'forwardPE',
        'fiveYearAvgDividendYield',
        'ask',
        'bid',
        'dividendYield',
        'fiveYearAvgDividendYield',
        'profitMargins',
        'forwardEps',
        'bookValue',
        'trailingEps',
        'priceToBook',
        'pegRatio',]

errorTicks = []
for ticker in tickers:
    sleep(1)
    if '-' in ticker:
        continue
    if '.' in ticker:
        continue
    try:
        tick = yf.Ticker(ticker)
        sleep(0.5)
        hist = tick.history(period="30d")
        variables = pd.DataFrame(data={
        'ticker':[ticker],
        'growth':(hist['Close'].values[-1])/(hist['Close'].values[0])-1,
        'price':(tick.info['ask']+tick.info['bid'])/2,
        'trailingAnnualDividendYield':tick.info['trailingAnnualDividendYield'],
        'payoutRatio':tick.info['payoutRatio'],
        'trailingAnnualDividendRate':tick.info['trailingAnnualDividendRate'],
        'dividendRate':tick.info['dividendRate'],
        'trailingPE':tick.info['trailingPE'],
        'priceToSalesTrailing12Months':tick.info['priceToSalesTrailing12Months'],
        'forwardPE':tick.info['forwardPE'],
        'fiveYearAvgDividendYield':tick.info['fiveYearAvgDividendYield'],
        'dividendYield':tick.info['dividendYield'],
        'fiveYearAvgDividendYield':tick.info['fiveYearAvgDividendYield'],
        'profitMargins':tick.info['profitMargins'],
        'forwardEps':tick.info['forwardEps'],
        'bookValue':tick.info['bookValue'],
        'trailingEps':tick.info['trailingEps'],
        'priceToBook':tick.info['priceToBook'],
        'earningsQuarterlyGrowth':tick.info['earningsQuarterlyGrowth'],
        'pegRatio':tick.info['pegRatio']
        })
        table = table.append(variables)
        print(errorTicks)
        print(table)
        
    except:
        print(ticker)
        errorTicks.append(ticker)
        print(errorTicks)

rotateTable = pd.read_csv("tableGrowth.csv")
rotateTable = rotateTable.drop('Unnamed: 0',1)
table.to_csv("tableGrowthDatabase.csv")
database = pd.read_csv("tableGrowthDatabase.csv")
database = database.drop('Unnamed: 0',1)
database = database.append(rotateTable)
database.to_csv("tableGrowthDatabase.csv")

