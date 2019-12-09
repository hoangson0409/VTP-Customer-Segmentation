# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 08:39:40 2019

@author: os_sonnh1
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming,euclidean
# from kmodes.kmodes import KModes
# from kmodes.kprototypes import KPrototypes
# import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode,iplot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
import seaborn as sns
import datetime, nltk, warnings
# init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
%matplotlib inline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion

df = pd.read_csv('VTP_test.csv',dtype={'msisdn': str})
df['request_date'] = pd.to_datetime(df['request_date'])

gdtc = ('641000','830000','654001','610000','300001','810000','860000','861000','640000','LX9001','BHV004', '640001','851000','654100','400100','641100','865000','VP2001','VP3001','VP6001','VP5002','VP3011', 'POS001','NAS002','NAS006','720000', 'QR0001', 'SSC02', '850000', '610301', '652000', '653000', '654000', '720003', 'QRCARD', 'CHILUO', 'TRATHU','EBK002','LNK003','LNK033','EBK002','LNK003','LNK033','TKBV02')
now = pd.to_datetime('2019-11-18 00:00:00')
dalkthe = pd.read_csv('tinhtranglkthe.csv',dtype={'msisdn': str})

def gender_preprocess(x):
    if x == 'MALE':
        return 0
    elif x == 'FEMALE':
        return 1
    else:
        return x
    
svcode_diennuoc = 'EVNHCM PA01GTNDH PA01HHNDH PA01MLNDH PC02CCQTI PC02DDQTI PC02EEQTI PC02FFQTI PC02HHQTI PC02KKQTI PC02LLQTI PC03AATTH PC03BBTTH PC03CCTTH PC03DDTTH PC03EETTH PC03FFTTH PC03GGTTH PC03HHTTH PC05AAQNM PC05BBQNM PC05CCQNM PC05DDQNM PC05EEQNM PC05FFQNM PC13CCDNC CPCBDH PA14BLCBG PA14TNCBG PA14TXCBG PA26BBBKN PA26BCBKN PC03TTTTH PC05IIQNM PC05KKQNM PC08FFPYN PC10MMGLI PC10NNGLI PC11BBKTM PC11CCKTM PC11EEKTM PC11FFKTM PC11GGKTM PC11HHKTM PC11IIKTM PC11KKKTM PC12EEDLK PC12GGDLK PC12LLDLK PC12PPDLK PC13DDDLK PC13EEDLK PC13FFDLK PC13GGDLK PC13HHDLK PC13IIDLK KHPCNT KHPCCRKS KHPCVNN TTHPC03 QNIPC06 KTMPC11 DLKPC12 PC13DDDCN PA14TLCBG PA15TXSLA PA15MCSLA PA15PYSLA PA15BYSLA PA15MLSLA PA15TCSLA PA15QNSLA PA15SMSLA PA15SCSLA PA15YCSLA PC05GGQNM PC05HHQNM PC05NNQNM PC12QQDLK PC13DLK PC13BBDLK PC13CCDLK KHPCCL PA01NDH PA02PTO PA04TNN PA05BGG PA07THA PA09TBH PA10YBI PA11LSN PC13FFDCN PC13GGDCN PC13IIDCN PC13HHDCN PA15SLA PA16HTH PA17HBH PA18LCI PA19DBN PA22BNH PA23HYN PA24HNM PA25VPC PA26BKN PA29LCU PMHDG PNNBH PA14BMCBG PA14HACBG PA14HLCBG PA14HQCBG PA14NBCBG PA14PHCBG PA14QHCBG PA14TACBG PA14TKCBG PC12KKDLK PC12MMDLK PC12NNDLK PA26CDBKN PA26CMBKN PA26NRBKN PA26NSBKN PA26PNBKN PA26PTBKN PC01AAQBH PC01BBQBH PC01CCQBH PC01EEQBH PC01FFQBH PC02AAQTI PC02BBQTI KHPCNH KHPCDKKV KHPCVNH PC13BBDCN PA0901TBH PA0903TBH PA0904TBH PA0905TBH PA0906TBH PA0907TBH PA0908TBH PB1101CTO PB1102CTO PB1103CTO PB1104CTO PB1105CTO PC06CCQNI PC06DDQNI PC06EEQNI CPCDCN CPCDNG CPCGLI CPCKHA CPCPYN CPCQBH CPCQNM CPCQTI PB1106CTO PB1107CTO PC13EEDCN HKMPD01 HBTPD02 BDHPD03 DDAPD04 TTIPD06 DAHPD08 SSNPD09 THOPD10 TXNPD11 CGYPD12 HMIPD13 LBNPD14 HDGPD16 STYPD17 CMYPD18 TTNPD20 BVIPD21 DPGPD22 HDCPD23 PXNPD25 PTOPD26 QOIPD27 UHAPD29 PC05LLQNM PP0100DNG PP0300DNG PP0500DNG PP0700DNG PP0900DNG PC06AAQNI PC06BBQNI PC06HHQNI PC06MMQNI PC06NNQNI PC06SSQNI PC06TTQNI PC07AABDH PC07BBBDH PC07DDBDH PC07EEBDH PC07FFBDH PC07GGBDH PC07HHBDH PC07IIBDH PC08AAPYN PC08BBPYN PC08CCPYN PC08DDPYN PC08EEPYN PC08GGPYN PC08HHPYN PC07CCBDH PC08IIPYN PC10AAGLI PC10BBGLI PC10CCGLI PC10DDGLI PC10EEGLI PC10FFGLI PC10GGGLI PC10HHGLI PC10IIGLI PC10KKGLI PC10LLGLI PC10OOGLI PC11AAKTM PC12AADLK PC12BBDLK PC12HHDLK PC12IIDLK PC12JJDLK PB1901BLU PB0407BDG PB0408BDG PA02TBPTO PA24HNHNM PA24KBHNM PA24LNHNM PA24TLHNM PA1901DBN PA1902DBN PA1903DBN PA1905DBN PA1906DBN PA1910DBN PA18BHLCI PA18BSLCI PA18BTLCI PA18BYLCI PA1101LSN PA1102LSN PA11BGLSN PA11BSLSN PA11CGLSN PA11CLLSN PA11DLLSN PA11HLLSN PA11LBLSN PA11TDLSN PA11VLLSN PA11VQLSN PA0501BGG PA0502BGG PA0504BGG PA0510BGG PB0404BDG PK01DNI PK05DNI PK07DNI PK08DNI PK09DNI PB1902BLU PA16CLHTH PA16CXHTH PA16DTHTH PA16HKHTH PA16HLHTH PA16HSHTH PA16HTHTH PA16KAHTH PA16LHHTH PA16NXHTH PA16THHTH PA25BXVPC PA25LTVPC PA25PYVPC PA25SLVPC PA25TDVPC PA25TMVPC PA25VTVPC PA25VYVPC PA25YLVPC PA0505BGG PB0401BDG PB0405BDG PB0410BDG PA01NDNDH PA01NHNDH PA0503BGG PA0508BGG PA0509BGG PB0409BDG PK06DNI PK10DNI PK11DNI PB1903BLU PB1904BLU PB1905BLU PB1906BLU PB1907BLU PB1001VLG PB1003VLG PB1004VLG PB1005VLG PB1006VLG PB1007VLG PB1008VLG PA23ATHYN PA01TNNDH PA01VBNDH PA01YYNDH BTLMPD30 PA23HYHYN PA23KCHYN PA23KDHYN PA23MVHYN PA23PTHYN PA23VGHYN PA23VLHYN PA23YMHYN PA02CKPTO PA02TNPTO PA13CCNAN PA13DYNAN PA13KSNAN PA13QLNAN PA13QPNAN PA13TANAN PA13THNAN PA13TKNAN PA0506BGG PB0402BDG PB0406BDG PA13CLNAN PA13NANAN PA13NDNAN PA13NHNAN PA13NLNAN PA13QCNAN PA13QHNAN PA13TCNAN PA13TDNAN PA13VHNAN PA13YTNAN PA12HYTQG PA12LBTQG PA12SDTQG PA12TXTQG PB1801NTN PB1802NTN PB1803NTN PA0507BGG PB0403BDG PK02DNI PK03DNI PK04DNI PA02DHPTO PA02LTPTO PA02PNPTO PA02PTPTO PA02TAPTO PA02TSPTO PA02TTPTO PA02VTPTO PA02YLPTO PA13ASNAN PA12NHTQG PB1804NTN PB1805NTN PA24BMHNM PA24DVHNM PA18CDLCI PA18SPLCI PB0110BPC PA1008YBI PA03BLQNH PA03BCQNH PB0508TNH PB0306LDG PB0307LDG PB0308LDG PB0310LDG PB0311LDG PB0312LDG PA03HLQNH PA03CPQNH PA03UBQNH PA03MCQNH PA03HHQNH PA03DHQNH PA03TYQNH PA18LCLCI PA18SILCI PB0101BPC PB0108BPC PA18VBLCI PA17CPHBH PA17KBHBH PA17KSHBH PA17LCHBH PA17TXHBH PA17YTHBH PM03HDG PM09HDG PM15HDG PM17HDG PA07NTTHA PA1001YBI PA17LSHBH PM21HDG PA1002YBI PA1005YBI PA2908LCU PA2906LCU PA20DVHGG PA20QBHGG PA20SPHGG PA20VXHGG PA20XMHGG PA20YMHGG PA20MVHGG PB0802TGG PB0803TGG PB0804TGG PM05HDG PM11HDG PM13HDG PM23HDG PA07TPTHA PA07SSTHA PA07BSTHA PA07HHTHA PA07NCTHA PA07VLTHA PA07NLTHA PA07CTTHA PA07NSTHA PA1007YBI PA2201BNH PA2202BNH PA20HGHGG PB0805TGG PB0807TGG PB0808TGG PA02HHPTO PA07HTTHA PA07HLTHA PA07QXTHA PA07QHTHA PA07NXTHA PA07TTTHA PA07BTTHA PA07THTHA PA07XTTHA PA07LCTHA PA07MLTHA PA1006YBI PB0809TGG PB0810TGG PB0501TNH PB0502TNH PB0103BPC PB0104BPC PB0106BPC PB0107BPC PB1806NTN PA12CHTQG PB0111BPC PA2203BNH PA2204BNH PA20BQHGG PA20BMHGG PA20QIHGG PB0503TNH PB0504TNH PB0505TNH PB0506TNH PB0507TNH PB0509TNH PB0301LDG PB0302LDG PB0303LDG PB0304LDG PA18MKLCI PB0102BPC PA1912DBN PA17DBHBH PA17LTHBH PA17MCHBH PA07QSTHA PB2001HUG PB2002HUG PB2003HUG PB2004HUG PB2005HUG PB2007HUG PB2008HUG PA1003YBI PA1004YBI PA2205BNH PA2206BNH PM01HDG PA07TGTHA PA07TSTHA PA07TXTHA PA07DSTHA PA07YDTHA PA2207BNH PA2208BNH PA2905LCU PA2901LCU PA2902LCU PA2904LCU PA2907LCU PNKS00NBH PB1309KGG PB1310KGG PB1605TVH PB0206BTN PB0702DTP PB0703DTP PB0602LAN PB0606LAN PB0603LAN PB0607LAN PB0612LAN PB0601LAN PB0611LAN PB0614LAN PB1201AGG PB1202AGG PH0900HPG PH0400HPG PH0200HPG PH0800HPG PH1200HPG PH1500HPG PB1302KGG PB1313KGG PB1602TVH PB1706STG PB1708STG PB1709STG PB1710STG PB0708DTP PB0709DTP PB0710DTP PB0712DTP PB0907BTE PB0908BTE PB1501VTU PA03QYQNH PNNQ00NBH PB1306KGG PB1312KGG PB1601TVH PB0704DTP PB0613LAN PB0605LAN PB1207AGG PB1507VTU PB1508VTU PA04DATNN PA04DHTNN PA04GTTNN PA04PBTNN PA04PLTNN PA04SCTNN PA04TPTNN PA04VNTNN PB1409CMU PNGV00NBH PNHL00NBH PNTD00NBH PB1303KGG PB1311KGG PB1604TVH PB1606TVH PB1405CMU PB1406CMU PB1408CMU PB0203BTN PB0205BTN PB1702STG PB1704STG PB1705STG PB0701DTP PB0706DTP PB0707DTP PB0610LAN PB1210AGG MLHPD15 TTTPD19 PA03VDQNH PA03HBQNH PA03DTQNH PNYM00NBH PB1301KGG PB1304KGG PB1307KGG PB1401CMU PB0901BTE PB0904BTE PB1502VTU PB1503VTU PB1504VTU PB1505VTU PB1506VTU PA04PYTNN MDCPD24 TOIPD28 PNYK00NBH PNNB00NBH PB1603TVH PB1607TVH PB1609TVH PB1410CMU PB0202BTN PB0207BTN PB1701STG PB0604LAN PB1205AGG PB1208AGG PB0902BTE PB0903BTE PB0905BTE PH0500HPG PH0600HPG PH1000HPG PH0100HPG PB1305KGG PB1608TVH PB1402CMU PB1403CMU PB1404CMU PB1407CMU PB0201BTN PB1711STG PB0705DTP PB0711DTP PB0609LAN PB0608LAN PB1203AGG PB1204AGG PB1206AGG PB1209AGG PB0906BTE PB0909BTE PH1100HPG PH1400HPG PH1300HPG PH0700HPG PH0300HPG PB0109BPC PQ05DKKHA PQ08KVKHA PB1109CTO PC01MMQBH PB0105BPC PP0800DNG PC05PPQNM PC02GGQTI PC12CCDLK PA16VQHTH PB0801TGG PB0806TGG PM07HDG PB0305LDG PA15MSSLA PA0902TBH PC11DDKTM PB0309LDG PA11TXLSN KHPCVH TLMPD05 PC06LLQNI PC03PPTTH PA1907DBN GLMPD07 PB1108CTO PC10PPGLI PA20HGG PC05MMQNM PA12YSTQG PB1002VLG PA14CBG PA1911DBN PA01NTNDH PA01XTNDH PM19HDG PA04DTTNN PC01DDQBH PB2006HUG PB1703STG PB1707STG PB1308KGG PB11CTO PC12FFDLK    NPHT NBT NTH NGD NTD NNB NCL NPCEXCEPT TUWACOLCU TPWACOLCU NTWACOQNM TKWACOQNM DBWACOQNM DNLMHPG NNT TBWACOQNM TDMBDGWACO DANBDGWACO TANBDGWACO KL2BDGWACO KL1BDGWACO TP1SLAWACO MCSLAWACO SMSLAWACO TP2SLAWACO MSSLAWACO QNSLAWACO TCSLAWACO PYSLAWACO LBHNI2WACO GLHNI2WACO DAHNI2WACO HAWACOQNM NTA'
svcode_diennuoc = svcode_diennuoc.split() + [
'NLC', 'QNMWACO', 'NDT', 'NCM', 'DNIWACO', 'NHCM', 'VIWACO', 'TGGWACO', 'STGWACO', 'VTUWACO', 'VPCWACO', 'NTDNIWACO', 'HADNIWACO', 'CTWACO', 'DNGWACO', 'SLAWACO', 'DHWACOKTM', 'BDGWACO', 'HNI2WACO', 'TNHWACO', 'TTHBNHWACO', 'HNI3WACO', 'HPGWACO', 'HPWACOHYN', 'VTSWACO', 'HBBGGWACO', 'THAHNIWACO', 'KONTUMWACO', 'HUDHNIWACO', 'LONGANWACO', 'BNWACO', 'NTWACO', 'NTNWACO', 'GLIWACO', 'BLUWACO', 'NHUE', 'LKDNIWACO', 'HGGWACO', 'QTWACO', 'NAMHNIWACO', 'TTHNIWACO', 'LTBNHWACO', 'STHNIWACO', 'BTNWACO'
]
svcode_taichinh = ['FECRDT', 'ATMCRDT', 'OCBCRDT', 'EASYCRDT', 'MSBCRDT']
svcode_muasam = ['IOGARENA','IOSOHACOIN','IOZING','IOGATE','CGVTMDT','CGV123','MOBILOTT','PHIM','123PHIM','MYGO']
svcode_vienthong = ['000000', '000001', '000002', '000004', '000022','100000','DATAVT','TELCO_VIETTEL','PCE','VNVNPPRE', 'VNVMSPRE', 'IOVNPPRE', 'IOVMSPRE', 'IOVNMPRE', 'VNVNMPRE',
       'IOBEEPRE']

def svcode_preprocess(x):
    if x in svcode_diennuoc:
        return 'DN'
    elif x in svcode_taichinh:
        return 'TC'
    elif x in svcode_muasam:
        return 'MS'
    elif x in svcode_vienthong:
        return 'VT'
    else:
        return x
    
class FeatureEngineering(BaseEstimator, TransformerMixin):

        
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        
        df['service_type'] = df.service_code.map(svcode_preprocess)
        #sum(gdtc_tc)
        df_temp1 = df[(df['process_code'].isin(gdtc)) & (df['error_code']=='00')].groupby('msisdn')['trans_amount'].sum()


        #count(gdtc_tc)
        df_temp2 = df[(df['process_code'].isin(gdtc)) & (df['error_code']=='00')].groupby('msisdn')['request_date'].count()


        #max_trans_amt(gdtc_tc)
        df_temp3 = df[(df['process_code'].isin(gdtc)) & (df['error_code']=='00')].groupby('msisdn')['trans_amount'].max()
        
        #last request date - recency
        df_temp4 = df[['msisdn','request_date']].groupby('msisdn')['request_date'].max()
        df_temp4 = pd.DataFrame(df_temp4)
        df_temp4['recency'] = round((now - df_temp4['request_date'])/np.timedelta64(1, 'D'))
        
        #last error code
        df_temp5 = df[['msisdn','request_date','error_code']].sort_values(by=['msisdn','request_date']).groupby('msisdn')['error_code'].agg(['last'])
        
        #pin_status
        df_temp6 = df[['msisdn','request_date','pin_status']].sort_values(by=['msisdn','request_date']).groupby('msisdn')['pin_status'].agg(['last'])

        #card_status
        df_temp7 = df[['msisdn']].drop_duplicates()
        df_temp7['card_status'] = df_temp7['msisdn'].isin(dalkthe['msisdn'].unique())
        df_temp7.card_status = df_temp7.card_status.astype(int)
        
        #hypothetical trans amount
        df_temp8 = df.groupby('msisdn')['trans_amount'].sum()
        
        #usage_length_in_days
        df_temp9 = df[['msisdn','request_date','process_code','error_code']].groupby('msisdn')['request_date'].max().reset_index() 
        df_temp91 = df[['msisdn','request_date','process_code','error_code']].groupby('msisdn')['request_date'].min()
        df_temp9 = df_temp9.merge(df_temp91,on='msisdn',how ='left')

        df_temp9['request_date_x'] = pd.to_datetime(df_temp9['request_date_x'])
        df_temp9['request_date_y']= pd.to_datetime(df_temp9['request_date_y'])
        df_temp9['usage_length_in_days']=round(( df_temp9['request_date_x']-df_temp9['request_date_y'])/np.timedelta64(1, 'D'))
        df_temp9 = df_temp9[['msisdn','usage_length_in_days']]
        
        
        #count_error
        df_temp10 = df[(df['error_code'] != '00') & (df['error_code'] != 'E51')].groupby('msisdn')['error_code'].count()
        df_temp11 = df.groupby('msisdn')['request_date'].count()
        
        #gender
        df_temp12 = df[['msisdn','request_date','gender']].sort_values(by=['msisdn','request_date']).groupby('msisdn')['gender'].agg(['last'])
        df_temp12.columns = ['gender']
        
        df_temp12['gender'] = df_temp12['gender'].map(gender_preprocess)
        
        #age
        df_temp13 = df[['msisdn','request_date','birthday']].sort_values(by=['msisdn','request_date']).groupby('msisdn')['birthday'].agg(['last'])
        df_temp13.columns = ['birthday']
        df_temp13.birthday = pd.to_datetime(df_temp13.birthday,errors ='coerce')
        df_temp13['Age'] = 2019 - df_temp13.birthday.dt.year
        df_temp13['Age'] = df_temp13['Age'].apply(lambda x: np.nan if (x > 75 or x <15) else x)
        df_temp13=df_temp13['Age']
        
        fin_df = df_temp4.merge(df_temp1,on='msisdn',how ='left')
        fin_df = fin_df.merge(df_temp2,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp3,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp5,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp6,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp7,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp8,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp9,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp10,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp11,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp12,on='msisdn',how='left')
        fin_df = fin_df.merge(df_temp13,on='msisdn',how='left')
        
        fin_df.columns = ['msisdn', 'last_request_date', 'recency', 'sum_gdtc',
       'count_gdtc', 'max_gdtc', 'last_error_code', 'pin_status', 'card_status',
       'hypothetical_trans_amount', 'usage_length_in_days','count_error','count_total_request', 'gender', 'age']
        
        #error features
        fin_df['error_per_day'] = fin_df['count_error'] / fin_df['usage_length_in_days']
        fin_df['error_req_percentage'] = fin_df['count_error']/fin_df['count_total_request']

        fin_df['last_error_sodu'] = fin_df['last_error_code'] == '16'
        fin_df['last_error_sodu'] = fin_df['last_error_sodu'].astype(int)
        fin_df['last_error_mapin'] = fin_df['last_error_code'].isin(['OTP','705','E09'])
        fin_df['last_error_mapin'] = fin_df['last_error_mapin'].astype(int)
        fin_df['last_error_cachdung'] = fin_df['last_error_code'].isin(['P45','03L','04L'])
        fin_df['last_error_cachdung'] = fin_df['last_error_cachdung'].astype(int)

        fin_df['last_error_code'] = np.logical_not(fin_df['last_error_code'].isin(['00','E51'])).astype(int)
        
        fin_df['tt_diennuoc'] = fin_df.msisdn.isin(df[df['service_type'] == 'DN']['msisdn'].unique()).astype(int)
        fin_df['tt_taichinh'] = fin_df.msisdn.isin(df[df['service_type'] == 'TC']['msisdn'].unique()).astype(int)
        fin_df['tt_muasam'] = fin_df.msisdn.isin(df[df['service_type'] == 'MS']['msisdn'].unique()).astype(int)
        fin_df['tt_vienthong'] = fin_df.msisdn.isin(df[df['service_type'] == 'VT']['msisdn'].unique()).astype(int)
        
        fin_df.rename(columns = {'last_error_code':'last_code=error'},inplace=True)
        fin_df = fin_df[['msisdn', 'sum_gdtc', 'count_gdtc', 'max_gdtc', 'recency',
       'hypothetical_trans_amount', 'usage_length_in_days','pin_status', 'card_status',
        'error_per_day','error_req_percentage', 'last_code=error',
       'last_error_sodu', 'last_error_mapin', 'last_error_cachdung',
       'tt_diennuoc', 'tt_taichinh', 'tt_muasam', 'tt_vienthong','gender','age']]
        
        fin_df[['sum_gdtc', 'count_gdtc', 'max_gdtc','error_per_day', 'error_req_percentage']] = fin_df[['sum_gdtc', 'count_gdtc', 'max_gdtc','error_per_day', 'error_req_percentage']].fillna(0)
        fin_df.pin_status = fin_df.pin_status.fillna('notset')
        fin_df.drop(columns=['age'],inplace = True)
        
        # fill gender null values with the mode of distribution
        fin_df.gender.fillna(0,inplace=True)
        fin_df.gender = pd.to_numeric(fin_df.gender)
        
        #convert pin status to one_hot format, then drop one columns to ensure no dummy variables trap
        fin_df['pin_lock'] = pd.get_dummies(fin_df['pin_status']).iloc[:,0]
        fin_df['pin_ok'] = pd.get_dummies(fin_df['pin_status']).iloc[:,1]
        fin_df.drop(columns=['pin_status'],inplace = True)
        
        fin_df['usage_length_in_days'][fin_df['usage_length_in_days'] == 0] = 1
        fin_df['error_per_day']=full_ft_df.count_error.fillna(0) / fin_df['usage_length_in_days']
        
        return fin_df

class DframeToArray(BaseEstimator, TransformerMixin):
    def __init__(self, col_idx):
        self.col_idx = col_idx
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:,self.attribute_names:].values
    
Pipeline([
        ('feature_engineer', FeatureEngineering()),
      
        ('df_to_array', DframeToArray(1)),
        ('std_scaler', StandardScaler()),
])