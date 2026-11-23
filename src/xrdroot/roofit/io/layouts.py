"""ROOT 6.40.04 workspace layouts and relocatable streamer descriptions.

Harvested from a native workspace by ``tools/roofit_write_schema.py``.
No ROOT installation is needed to use these descriptions. Each member tuple
is its name, ROOT streamer type, C++ type and fixed array length.
"""

from __future__ import annotations

import base64
import json
import zlib
from functools import cache

#: Class versions and their persisted members, in declaration order.
LAYOUTS: dict[str, tuple[int, tuple[tuple[str, int, str, int], ...]]] = {'TNamed': (1,
            (('TObject', 66, 'BASE', 0),
             ('fName', 65, 'TString', 0),
             ('fTitle', 65, 'TString', 0))),
 'TObject': (1, (('fUniqueID', 13, 'unsigned int', 0), ('fBits', 15, 'unsigned int', 0))),
 'RooWorkspace': (8,
                  (('TNamed', 67, 'BASE', 0),
                   ('_uuid', 62, 'TUUID', 0),
                   ('_classes', 61, 'RooWorkspace::CodeRepo', 0),
                   ('_allOwnedNodes', 61, 'RooArgSet', 0),
                   ('_dataList', 61, 'RooLinkedList', 0),
                   ('_embeddedDataList', 61, 'RooLinkedList', 0),
                   ('_views', 61, 'RooLinkedList', 0),
                   ('_snapshots', 61, 'RooLinkedList', 0),
                   ('_genObjects', 61, 'RooLinkedList', 0),
                   ('_studyMods', 61, 'RooLinkedList', 0),
                   ('_namedSets', 500, 'map<string,RooArgSet>', 0),
                   ('_eocache', 61, 'RooExpensiveObjectCache', 0))),
 'pair<string,RooArgSet>': (1, (('first', 500, 'string', 0), ('second', 61, 'RooArgSet', 0))),
 'RooArgSet': (1, (('RooAbsCollection', 0, 'BASE', 0),)),
 'RooAbsCollection': (3,
                      (('TObject', 66, 'BASE', 0),
                       ('RooPrintable', 0, 'BASE', 0),
                       ('_list', 500, 'vector<RooAbsArg*>', 0),
                       ('_ownCont', 18, 'bool', 0),
                       ('_name', 65, 'TString', 0),
                       ('_allRRV', 18, 'bool', 0))),
 'RooPrintable': (1, ()),
 'TUUID': (1,
           (('fTimeLow', 13, 'unsigned int', 0),
            ('fTimeMid', 12, 'unsigned short', 0),
            ('fTimeHiAndVersion', 12, 'unsigned short', 0),
            ('fClockSeqHiAndReserved', 11, 'unsigned char', 0),
            ('fClockSeqLow', 11, 'unsigned char', 0),
            ('fNode', 31, 'unsigned char', 6))),
 'RooAddPdf': (5,
               (('RooAbsPdf', 0, 'BASE', 0),
                ('_refCoefNorm', 61, 'RooCollectionProxy<RooArgSet>', 0),
                ('_refCoefRangeName', 64, 'TNamed*', 0),
                ('_pdfList', 61, 'RooCollectionProxy<RooArgList>', 0),
                ('_coefList', 61, 'RooCollectionProxy<RooArgList>', 0),
                ('_haveLastCoef', 18, 'bool', 0),
                ('_allExtendable', 18, 'bool', 0),
                ('_recursive', 18, 'bool', 0))),
 'RooAbsPdf': (5,
               (('RooAbsReal', 0, 'BASE', 0),
                ('_rawValue', 8, 'double', 0),
                ('_errorCount', 3, 'int', 0),
                ('_traceCount', 3, 'int', 0),
                ('_negCount', 3, 'int', 0),
                ('_selectComp', 18, 'bool', 0),
                ('_normRange', 65, 'TString', 0))),
 'RooAbsReal': (3,
                (('RooAbsArg', 0, 'BASE', 0),
                 ('_plotMin', 8, 'double', 0),
                 ('_plotMax', 8, 'double', 0),
                 ('_plotBins', 3, 'int', 0),
                 ('_value', 8, 'double', 0),
                 ('_unit', 65, 'TString', 0),
                 ('_label', 65, 'TString', 0),
                 ('_forceNumInt', 18, 'bool', 0),
                 ('_specIntegratorConfig', 64, 'RooNumIntConfig*', 0))),
 'RooAbsArg': (9,
               (('TNamed', 67, 'BASE', 0),
                ('RooPrintable', 0, 'BASE', 0),
                ('_serverList', 62, 'RooSTLRefCountList<RooAbsArg>', 0),
                ('_clientList', 62, 'RooSTLRefCountList<RooAbsArg>', 0),
                ('_clientListShape', 62, 'RooSTLRefCountList<RooAbsArg>', 0),
                ('_clientListValue', 62, 'RooSTLRefCountList<RooAbsArg>', 0),
                ('_proxyList', 61, 'RooRefArray', 0),
                ('_boolAttrib', 500, 'set<string>', 0),
                ('_stringAttrib', 500, 'map<string,string>', 0),
                ('_valueDirty', 18, 'bool', 0),
                ('_shapeDirty', 18, 'bool', 0),
                ('_operMode', 3, 'RooAbsArg::OperMode', 0),
                ('_fast', 18, 'bool', 0))),
 'RooSTLRefCountList<RooAbsArg>': (3,
                                   (('_storage', 500, 'vector<RooAbsArg*>', 0),
                                    ('_refCount', 500, 'vector<unsigned int>', 0))),
 'RooGaussian': (1,
                 (('RooAbsPdf', 0, 'BASE', 0),
                  ('x', 61, 'RooTemplateProxy<RooAbsReal>', 0),
                  ('mean', 61, 'RooTemplateProxy<RooAbsReal>', 0),
                  ('sigma', 61, 'RooTemplateProxy<RooAbsReal>', 0))),
 'RooAbsRealLValue': (1, (('RooAbsReal', 0, 'BASE', 0), ('RooAbsLValue', 0, 'BASE', 0))),
 'RooAbsLValue': (1, ()),
 'RooChebychev': (2,
                  (('RooAbsPdf', 0, 'BASE', 0),
                   ('_x', 61, 'RooTemplateProxy<RooAbsReal>', 0),
                   ('_coefList', 61, 'RooCollectionProxy<RooArgList>', 0),
                   ('_refRangeName', 64, 'TNamed*', 0))),
 'RooRecursiveFraction': (1,
                          (('RooAbsReal', 0, 'BASE', 0),
                           ('_list', 61, 'RooCollectionProxy<RooArgList>', 0))),
 'RooConstVar': (2, (('RooAbsReal', 0, 'BASE', 0),)),
 'TProcessID': (1, (('TNamed', 67, 'BASE', 0),)),
 'RooNumIntConfig': (1,
                     (('TObject', 66, 'BASE', 0),
                      ('RooPrintable', 0, 'BASE', 0),
                      ('_epsAbs', 8, 'double', 0),
                      ('_epsRel', 8, 'double', 0),
                      ('_printEvalCounter', 18, 'bool', 0),
                      ('_method1D', 61, 'RooCategory', 0),
                      ('_method2D', 61, 'RooCategory', 0),
                      ('_methodND', 61, 'RooCategory', 0),
                      ('_method1DOpen', 61, 'RooCategory', 0),
                      ('_method2DOpen', 61, 'RooCategory', 0),
                      ('_methodNDOpen', 61, 'RooCategory', 0),
                      ('_configSets', 61, 'RooLinkedList', 0))),
 'RooCategory': (3,
                 (('RooAbsCategoryLValue', 0, 'BASE', 0),
                  ('_rangesPointerForIO', 500, 'map<string,vector<int> >*', 0))),
 'RooAbsCategoryLValue': (1, (('RooAbsCategory', 0, 'BASE', 0), ('RooAbsLValue', 0, 'BASE', 0))),
 'RooAbsCategory': (4,
                    (('RooAbsArg', 0, 'BASE', 0),
                     ('_currentIndex', 3, 'int', 0),
                     ('_stateNames', 500, 'map<string,int>', 0),
                     ('_insertionOrder', 500, 'vector<string>', 0))),
 'pair<string,vector<int> >': (1, (('first', 500, 'string', 0), ('second', 500, 'vector<int>', 0))),
 'pair<string,int>': (1, (('first', 500, 'string', 0), ('second', 3, 'int', 0))),
 'RooLinkedList': (3,
                   (('TObject', 66, 'BASE', 0),
                    ('_hashThresh', 3, 'int', 0),
                    ('_size', 3, 'int', 0),
                    ('_name', 65, 'TString', 0))),
 'TString': (2, ()),
 'RooCollectionProxy<RooArgList>': (1,
                                    (('RooArgList', 0, 'BASE', 0),
                                     ('RooAbsProxy', 0, 'BASE', 0),
                                     ('_owner', 64, 'RooAbsArg*', 0),
                                     ('_defValueServer', 18, 'bool', 0),
                                     ('_defShapeServer', 18, 'bool', 0))),
 'RooArgList': (1, (('RooAbsCollection', 0, 'BASE', 0),)),
 'RooAbsProxy': (1, ()),
 'RooCollectionProxy<RooArgSet>': (1,
                                   (('RooArgSet', 0, 'BASE', 0),
                                    ('RooAbsProxy', 0, 'BASE', 0),
                                    ('_owner', 64, 'RooAbsArg*', 0),
                                    ('_defValueServer', 18, 'bool', 0),
                                    ('_defShapeServer', 18, 'bool', 0))),
 'RooUniformBinning': (1,
                       (('RooAbsBinning', 0, 'BASE', 0),
                        ('_xlo', 8, 'double', 0),
                        ('_xhi', 8, 'double', 0),
                        ('_nbins', 3, 'int', 0),
                        ('_binw', 8, 'double', 0))),
 'RooAbsBinning': (2, (('TNamed', 67, 'BASE', 0), ('RooPrintable', 0, 'BASE', 0))),
 'RooRealVarSharedProperties': (2,
                                (('RooSharedProperties', 0, 'BASE', 0),
                                 ('_altBinning', 500, 'unordered_map<string,RooAbsBinning*>', 0))),
 'pair<string,RooAbsBinning*>': (1,
                                 (('first', 500, 'string', 0),
                                  ('second', 64, 'RooAbsBinning*', 0))),
 'RooSharedProperties': (1, (('TObject', 66, 'BASE', 0), ('_uuid', 62, 'TUUID', 0))),
 'RooTemplateProxy<RooAbsReal>': (1, (('RooArgProxy', 0, 'BASE', 0),)),
 'RooArgProxy': (1,
                 (('TNamed', 67, 'BASE', 0),
                  ('RooAbsProxy', 0, 'BASE', 0),
                  ('_owner', 64, 'RooAbsArg*', 0),
                  ('_arg', 64, 'RooAbsArg*', 0),
                  ('_valueServer', 18, 'bool', 0),
                  ('_shapeServer', 18, 'bool', 0),
                  ('_isFund', 18, 'bool', 0),
                  ('_ownArg', 18, 'bool', 0))),
 'RooPolynomial': (1,
                   (('RooAbsPdf', 0, 'BASE', 0),
                    ('_x', 61, 'RooTemplateProxy<RooAbsReal>', 0),
                    ('_coefList', 61, 'RooCollectionProxy<RooArgList>', 0),
                    ('_lowestOrder', 3, 'int', 0))),
 'RooBinning': (3,
                (('RooAbsBinning', 0, 'BASE', 0),
                 ('_xlo', 8, 'double', 0),
                 ('_xhi', 8, 'double', 0),
                 ('_ownBoundLo', 18, 'bool', 0),
                 ('_ownBoundHi', 18, 'bool', 0),
                 ('_nbins', 3, 'int', 0),
                 ('_boundaries', 500, 'vector<double>', 0))),
 'RooRangeBinning': (1, (('RooAbsBinning', 0, 'BASE', 0), ('_range', 28, 'double', 2))),
 'RooDataSet': (2,
                (('RooAbsData', 0, 'BASE', 0),
                 ('RooDirItem', 0, 'BASE', 0),
                 ('_varsNoWgt', 61, 'RooArgSet', 0),
                 ('_wgtVar', 64, 'RooRealVar*', 0))),
 'RooAbsData': (7,
                (('TNamed', 67, 'BASE', 0),
                 ('RooPrintable', 0, 'BASE', 0),
                 ('storageType', 3, 'RooAbsData::StorageType', 0),
                 ('_vars', 61, 'RooArgSet', 0),
                 ('_dstore', 64, 'RooAbsDataStore*', 0),
                 ('_ownedComponents', 500, 'map<string,RooAbsData*>', 0),
                 ('_globalObservables', 64, 'RooArgSet*', 0))),
 'RooDirItem': (1, ()),
 'RooVectorDataStore': (7,
                        (('RooAbsDataStore', 0, 'BASE', 0),
                         ('_varsww', 61, 'RooArgSet', 0),
                         ('_wgtVar', 64, 'RooRealVar*', 0),
                         ('_realStoreList', 500, 'vector<RooVectorDataStore::RealVector*>', 0),
                         ('_realfStoreList', 500, 'vector<RooVectorDataStore::RealFullVector*>', 0),
                         ('_catStoreList', 500, 'vector<RooVectorDataStore::CatVector*>', 0),
                         ('_sumWeight', 8, 'double', 0),
                         ('_sumWeightCarry', 8, 'double', 0),
                         ('_currentWeightIndex', 17, 'ULong64_t', 0))),
 'RooAbsDataStore': (1,
                     (('TNamed', 67, 'BASE', 0),
                      ('RooPrintable', 0, 'BASE', 0),
                      ('_vars', 61, 'RooArgSet', 0),
                      ('_cachedVars', 61, 'RooArgSet', 0),
                      ('_doDirtyProp', 18, 'bool', 0))),
 'RooRealVar': (10,
                (('RooAbsRealLValue', 0, 'BASE', 0),
                 ('_error', 8, 'double', 0),
                 ('_asymErrLo', 8, 'double', 0),
                 ('_asymErrHi', 8, 'double', 0),
                 ('_binning', 64, 'RooAbsBinning*', 0))),
 'RooVectorDataStore::RealVector': (1,
                                    (('_vec', 500, 'vector<double>', 0),
                                     ('_nativeReal', 64, 'RooAbsReal*', 0),
                                     ('_real', 64, 'RooAbsReal*', 0),
                                     ('_tracker', 64, 'RooChangeTracker*', 0))),
 'RooChangeTracker': (1,
                      (('RooAbsReal', 0, 'BASE', 0),
                       ('_realSet', 61, 'RooCollectionProxy<RooArgList>', 0),
                       ('_catSet', 61, 'RooCollectionProxy<RooArgList>', 0),
                       ('_realRef', 500, 'vector<double>', 0),
                       ('_catRef', 500, 'vector<int>', 0),
                       ('_checkVal', 18, 'bool', 0))),
 'RooExpensiveObjectCache': (2,
                             (('TObject', 66, 'BASE', 0),
                              ('_nextUID', 3, 'int', 0),
                              ('_map',
                               500,
                               'map<TString,RooExpensiveObjectCache::ExpensiveObject*>',
                               0))),
 'RooRefArray': (1, ()),
 'RooWorkspace::CodeRepo': (2, ())}

#: Native streamer entries, compressed together; class references are relocated.
ENTRIES = (
    'c%1FM=aTYB(<Xc>%U}Y;5FvqZixu7*0dgA)NO&*2=h(lhlJKIt@1A-0d!N|l$34^7tgNieD{cL+|Mge3AWk~%zy2ov'
    '{wq7-eblM2jN-Is8``i_S+a<_WNasFM28a}m3JBN7SjdCy)HYhJiIQ#ZV%TeAvVpfY@toM2A?Is`)q>xRk<^rvp4Zs'
    'D>(Ng@J>b7MIXZ-A6^Ii*MQ?>f(0M*UdmXA3@~MRddyX%W*c`|2Zq|PlnCRvgTM*K)3ij>l(UV~wD<Z|rY_Db<c>iq'
    'S$$o}U5`Y{t+^YcM7Pe(PBH9&<MxUPe|r!+Pn~jjpPwXiC1O;$F2?f47k%Ds=YsjM9=vL{;dque;ogMX3I9wyatMBd'
    'u0NoK*r<F~=%ZVu7(2Vn-E7!sJ;8Sv;XlQ9(%v5ST4pJTMp<-2s$5@EWmc&2=q+#w=Ew_GcG7C=Ny4~)63n4?h&dj8'
    '1uhs)IDbhHj{o(){{{k8l!l$w>MBvTFC<FJZlOe(@mGneWqL6_IS-xi1RF2J+g0Nb=c5Z#YBRJhPtLO0#YauS9YfoE'
    'G>%hM2|pfd<O^ZwU&y8ED+LEBHRl3u3KOZGJr5+xTEGu&oryIK;BFA>F%YZhomE~oxYX&y_Sj~~1&h8BB8?Jt(;|GX'
    'zy*@#kR)Vl4pwFWr371#;ka0QAxPsP4TK&y^a!0$&A|l=_7#x3Nnhgt^qKlp7h>C)Qxm65(KO^*0LNP+3r^;(&j%`*'
    '&s9q!ohy(G;gk@m;$ROp35n;5ISG$!CDCY)4RAtsCVoXE5klKc&t~jwF<Q!EtMk|29=O+nWA$;b?d6=)0I7Dq?6u>#'
    'APyDJl4?kv_4v%T%d>N$ks6e6JYe<Nb89;gpb1%L7PC}eG#D~r@I&@Bj??K^(j7i)A&NA`a9JeRKu#JJ(HP$xS(&F='
    'lDky}UrMHV!eO$R1-oGnMAn|vJ@2)P_Cx~}b{Z=9xY4oPWawPX4cIu&Q2c~X!W}VRHHy^}y$m}^wjVyV(8-A<T-9!F'
    '$au`JOe_TQt4uLc+dn?pTFlA@>ju0zQZeFDOP7aBJxOBn=l}{;uUFJX?PT$xWmP8?s~@bBB(xRaL^qanW0s&>!GT}W'
    'd3IF+lb3ms^5!s#B~A`yy*LqJzuk4*Nqd3~U#`*Djd!ifJpRPa>0CLUkSbZUJHUQ8TNt9MBC{RmBUQmB%Uf+c#mfCm'
    'iIr2+IN((=*J_ZCG9}mroSoJyIyO;dFM!#-40CLSY;h|UiEu%q5M}6U!w$GH39~X=w+o44ovlsum<`HyO!AW40eAJw'
    '2R!1bWxidNIU`_&eN!}I`T|oL1`_GR)_P<oxMwXGG9~AzP|t5aF13|y?!c)b@d{dMMsD?V4@)J6jL6$Ap#VH9Q^Deu'
    '?9^@}2Y|xw=P1$BVcBW0;bikz$jg-X@`X9JGcEy3MBS2@7u9vd)@1+NbyUE|@{I$R(z8(4TS{*y&l=kxqX%Cl6iWa!'
    'M;4yJ;)^A45M*W~@r-{dd7X&>7qqYy!BUDOTkPTDr1)^%33i4c#p5Pf1B-4Sgv&lae?uZ6*hIlN3h4NL(=mWw3WVPo'
    'XuhboQh(<fFvsQ|P(@zWND=orvdsZuH8-9QXR5ZQ+<HY`G6{c|DY&_XRE_dZK=_Dfv5UQk^c0&fPOXx=*UnSoqT~e|'
    'KxEk7dg5RfF82=XLj5do!*HCfo10@*e46#|R)pgn3WyJ_7rH~NUL1EPKFHn7SG+ejRo+JqarkHY%?f))7%LxBarAva'
    'wfOTwe0e<b`cSw4UPyv3u*U#)ah$%V50905j&27)fc=@xI`(U+hFPgW4PVSZ3bjZ(oB;ed%y}f#NiHwVABEcMo(Y{!'
    '-^)#$c+O)qoGhw5CyfcXgJ*ay^Y^#F4;|nKe?Try7Mhd}teo4j({Q&Zd0p!c9_L3>O-)!NCKMjgnP|w#vts|86gPdc'
    'e|(P1tWOK&6JjA3yvef%fG#(|1)N2<`s(e_{eJh^FHfFF+5O01iD+hCPhN%bA0?gL7zEm)u8M%rl9J$j#6qpJ?O<d|'
    'r&y2@f@D<E1(wNw&fUb8J3MpOOuIwHC3w5oQEcOet9vzY5LMf_N;dFSx*=q2h$)9^Q5HL22=FaAI~2XDXk_Vw!q`Qv'
    '7R2r2-S;rkFl5T%>Oc}xo-7xe9C5+G3X8I(<jkXGphZJ=w!dU{LSc5a5aA)SE3ZdwtBdWK#xbyCdlW7)&AoVJ>%u<6'
    '&YcM_1UUtuBkFc^H14`CA(q{fZx&$hjz)sP67~TJo4Q)0)Lx@+$*l}snr){LWRk_f!oOTC2Gic(06tNRUZt}NUV*ap'
    '^cg4%KBMc+v7plqcN*Uv#l1x^A_N}KS_Mze-k1Y3JNFhHs{_lxcS87Yb!oa|mG~z5?v5$H0o^gN{i3eRrgkf>Y?LRE'
    'I05kOM|#bQ<JFEC_oSsG{L}sB0u4GgV5|>IV8AynfR;BY@U%1dt!!L5?8<~9$hE=oXrl!FX_8PUmcKA34!0edld|2a'
    '1B)T-)fcQMxx0Qs=op;b6IJMd{fw&1dbd?MJ`G=iv9hz*jm18rV!RNtAyjOC^?)9~x3Tu-XT=71#R1$h*C2+L3>WGV'
    'O=CHSwD`h4pPe6J*Y&Fak$D52aGX}phknM;*~=Clo2asNR~^-P**a%7BpGu=ChQa}r<{X1UL=>m30VX0!#5$sGi0^U'
    '@n@DJ+nkG*Bjo)QX>!gh?Y8Yys3AQ*c?ZOF`o>IY8J&&Cz@;3_rhonIKs*h<`E)m)5}uu>1=rzy;i=|2YF;zOoMx1a'
    'ctAKdM&p~i!H3FP?gmj0FI|ZDQhm{9wcQN|Lm`z4HYR2PpQ*N<MbM27B-CJa#n1B-8>!Tbw`Ijb6)q(vmv&s<!WMVO'
    'Sd)Lyp@`47@~1ubsC(|I6(Fy~SGxqgvXbvmz$BhsHrU>%x0iUV=K+}6A3#b|jMO(t_0=b(skz;F%<0vavEY-{`i|V2'
    'U2@eB6M-W2T_CXRJTuW*Tl=N6RL*-wk~px(?*5tN^;8*PL8Bu89`{h$_<j)7NVmuqFbLT6L_?H4lpHJDCN{-w8RbN|'
    '^oJgv#AU-9cXQDY9o0*9Bo^&dm;i8(^fCxo<r8?-`s^4DTI}9r7J8kLgST2wxFq;WMK-@*+qr<S%ra??vqNOt=0+3T'
    'D_EnR8bAll3dLj9DA<o$S|X-W2#8`K{jS9|w;?59#TMYH!(VvS?BM_ELf3f&HrIZ-AkA5F1_Y=KFO^r0BP7sC#bj}V'
    '<zqr@T!EUVWc(qE-RS57eLm>{4$<OwF%SDHqK33k%!3K_p4OXBG3B0NRc=?hXWn+sx`9)*i3z=4lVj*egDkfiLx7|4'
    'C14W_Ab$HGnPoKTP(xZk3zdM?UWwAUbJgnR%V(|b@dZ2|V;v8ts+}?nCZ>4PqZ|M-Z*9LT9~eq_S3cK=5^}56(y=RH'
    'b@yl0m=MNE7{Wcj5Jv20J+d$wP9vp)I{=Rfv^1=9sbOi%K-<m5H%qe=U5cgA)H+Y+La-Z_fHjU)8l4M(uc+zc_l1&0'
    'E*KMK!Rqy{5^aO60vwKQ706hg;eC~1T-=yrVT#{7BS_(pi6|Q*iUskzSQb&iG09yodK3q+*Nq5x++7)cO)e+KVQkpZ'
    'i6L@h2^ZVbAa~9MV!=8WNE-ZXZ@D)C$Fq_Ls<Cxc&@fT(0YjU9I7FWw{n0((Cb_X*203xEzUmCV8JT)S-?mTb2Gr@a'
    '0)Jzp^9ah#^wo~fs`x^PP4eTMy`Rakvfu~)Ote|#0s}9Zu$%&6xDMbN^_pI)VwnUet}JGAq3GHmn%2iG@R$YX|4)mV'
    'y)4F^tTzUN1)uC$xs;dTic+jRkO1~>rQ+OJ@P?@Bxi;o~v=v99BH^C%Jn7}h8t`4F()G<r01R<1g8xh{#3a3^kms_C'
    'VgTr1cUU|m=ANGDzKp(F40Kgd44~CW!9R1a*7mFm@YlughgN<!i?@obA7hSDSC|*49=po+xGC~8J>MDL*$y0?QULNa'
    '^=%-Wk07;QsqeOaa9&dUi`r)cp@PhItXo#-o}4FOjF>pHO1qSU=sgCD41z@pHiCyl`M;M3A-Sx#gBa#U<NE#v_uB1J'
    'yZhJQPNdAv{AXpRUk&=7u`?kJoNV4`ZFgeL=@z__z;Jxr2y19GELBTeH*XrvH}R<0Ip$b@!cj35M=kahNK5zC<y2_N'
    'gwt%^oRXR4WR+%0Bc3do&BkVnO~o^maG_`+7Rxo-l~O$cPUNm4>C0tWH0g>@;o=8@(OZwyCY^qIGHT6U=Bcr|)T^7I'
    '?HM^20kqN6xS5dwnf~MRMn`L#-ocmK<VM|6${6>hOqC1{GzLI2&6?Q()<HR1B;h-<y^Rxw*OEDLV+kbLS!85onpi&w'
    'lUxM|8PlRM0gnEoYeoFT>+^~ZfQdIb^AaMtqkzS^pkU37d=FuU+M?+LT8Ox4MvK@<$Oj-mJBifzpoZ^xVAwZfq{hp&'
    '=;6^{B7D;|JEO{7@Q<x0biKnl<XXDiIkiV%36Tp}s-1){xP(rJ3#&N+tm<IHErW#%l#_079&`o3CPR{y7mSk&G69<+'
    'LpWaae7|P*E%`8qewXg6P-(4&c2JVb;{+goJb2zgPi<2iVJM*Fj>jdv60-y3M#PA`^a9L@pL5Mr=B!2+o$DWLN)7U!'
    'd2pXvYbMw_q=dd2jmGUeWmDdROphVF7_n%htf~s<W{$y=TC2b$K*NuycMXP_!;Sg}X5S+;vPc}iqu!BqECj=>wc%}f'
    '?*;5k&?K62scwdCc#F`}PQx%EI@A8|)TEO&Bi@hd;{_nx(T_j#K2}?cfdD(M4W|;5IIq}1mX9)}#{ML9%<TX}*D?++'
    'TB_SA!Pt#Tw8VGwOL=z?Ea_ng{E*)Ywug`yIz7oiRRAn>2|IJatg`@fQs68kmOwoOf;j;jTUWn<|IS_T8^|3`@-xVd'
    'oU%ur;p+JG8~q}L+`ZO*JIJ&7LQ~HUOo5Q*DtX!Cq*!CS`6z!^XwJ@^bPRkVVdQP;S^HOu8QZ>DOw34)kOfp1<7L86'
    'zrkdI={C3^%%MN6BLPnQ+NsEWuhxu<#wc3=AKN-$RtNf<0T(c8Se*CQ@ZF!TLkd0wJRU4yL-=d~>CR~Ig#|=b$7r3~'
    '?Cy{SSImM?uWMm?-;s;&-z=cM=T<R&k{^GvfTUspsf`Qe^4V8xQ$AaEkd`c<o@>JYTZ#sKhoD4(Bza@eTH$>s8vJQX'
    '<~V<#V*zf_KB>v5a=mcz*Q@M@Bq}n*<zDt{I1gOf>JBVx<U@AA%~d+ZM>~KM6NX+Jc7Upq4L;+!{qxPkPA*mrW`WHa'
    'pq`J*Z$vWEtXCoba783teD?5?#sG%Inh*~oBL>ug_af=eG7h}$5`YZ$894fKm<5K(a|B-W%E?8`<vb;(n1ax2`B8q{'
    '@w5CsRZu;lMriqs<968}0V5oSoRa=5M!3Tx>(a3~*xI%&x5;~r5^Wf}gjn8nLeW9+r5M4=-AkUG1^hmtTfrT`MH-4q'
    '90|s`@Z*to{KSvLYt0FABc?qsZTZrT$f_v;OBWg(^Wr`8Io)$fhr9(W-X<Pdd^Wn^(T@NZ?wAqGp{Riil6@~QJ+;`s'
    '$@b*RZvC3=-3YdV7N3?7+l!v?IrPkSnVlIM7_>Ed(dn|_Lfjyx*x5zSNbX(&S3|bLF)~680h^pIN?^;G<2UKOE(1)8'
    'rDhc3`l1znB?6W|aEEY$H<d8noIl&!zEQOGv=xVdQWgZ4z+jU}6R^AFQG94@KN~#f7>7WfcOj;{KiFI;v15Kl#58&V'
    'zV-KoNs4NvJL32nMLx|w>(+H~xq}kpc#Yva=`;E0G8T47lE9mfHmw{BF%5FzcS}H{f-{(7$?+dc-k5lUYoJ7iQt{z4'
    '6uOUU`Z8t(1LK<l03#0`b?f)aG|e_*o{jAP5RpBH{J`4{cfJ-)rk86&`tokgeZ_;#A@8dQJ}#J9Z{dL<CEYP!WwDDh'
    'E?{PgyY1`mv~0WcUEr|QNT?Nhnn<&h79+)WzUV&|_TZo5PDm7YG)xw!AmKD%r|*K8XrGLH9`w(56mHIA>43#AC?)x='
    '%U$s*E_cz{0Ynnr-b?Kn!0|SK1zo$1c-EBzca4e&hoKFp5_bP%?b+Wm!t6`!X;<SG!E{c`w!_+m1F)0jjFRW<tz{Q<'
    '1M>h_=;Hvh8w&8%G^!=J=QI8#PCnH-hSUnoy+px3v#;sLgNX-17@+*X@T3YTfuUACUQGEGX-_B6CTD|(pwYcwjlR_|'
    'rY@;uK@%J4)NMyL{g2A;C4l%A@7!40Oi8pDyq<pB2e`(^n1!o$`L~G#-b!cx<#tEaVufa%flM5{g{knlQ8XMkb4hpF'
    'V^SqCI>J8z9G0;tSg$?E>qD&s@cO-8Xgf(@bJkt}o23h|lZ6P{eWytmPQO7-tw-C~7c9+91<%Hr^bb7zS=)I42HsgU'
    'AwE|p6|?8zhwKIaL8OMmA^e0>G#Fel2(EN&*chTqX&xof?GzU*INV&0l$^j$o|C}WN;tS2n{%m|&OtxRG_jG5@!L5P'
    'V1-T@mZ=r);td1rhM$Nxp(opd(I^EAvU6Ehekevi`T^9U_t)QER9oqee6FqB*zoL^et<(XR=H`(ZRxeZE3_ZS0~xK@'
    'zRz=MxR}??QR-R$X(UXo63<0xs%|qe^Hn7%;SqNhC=7EJMO4^C6(f||+R7kOMDc#WbYGV3Pvs5nj_TgZDsgw+RPf{O'
    '>C=<<NEl3GkUsgNeHE9rR0O<jn?WLzrW3C1BQ7Z$1;Kdk0Drt!KJvEAZi~c?4b_u7B><#AKjl?@^r5Us`$1OJgwIEE'
    'oq}0A>~kj6nvGb+)>Ao-3ZMvj^5QuF)W;RT0q?9OuM;4FYXMXixD+QZ1h|QZBA2>xq66lq;`l~h9n@(qn2L)yjEO}B'
    'tbVMb2#pd#QN)ayI{uv{Fvp1B4-oRi{JaE`w{6=O7!OMQS1!SI{!mzQTSD4?M4bT}WeLFYxSy6VEk7>7U9B_ax|xx@'
    '$s^bW*_marvrM2s7B~h|nxW}shLy{HH+b{K4fIULBU1gilM4d$d)1+F;P!Emv!YyUu65}@VCAg<1OelpuiTbQz~W`<'
    '|Mbd9uJu(hQ9q7=S~a-FLr^KU--_c(q`P%UollaNY{qXAn}Tu15-WFXBc;|5DxPm>1}DHHV1=QP_4nH=h7NdXjUddm'
    'KElx`qxuNoEr`I&A_o0u(jv8$xf$fmVjP&b*}0jkHq2{aL^?QN=RD2%FTl-3^!>bKkMFYboSduhqFrh#DPY%d+R-RB'
    'x>a(h)%TpfTXbkdtz)&#ypn)r0ewjn#${&U8SM~~qIj9jf&at30ABRNe^kx^d`Ye)c|EO?&~*>kN1CChj%ZG|v-Yoj'
    'ospbv0joyVmzwYbxZz8t+;^RG=f+mwIvJg1;B6n=P~_vl;VF0(EZhlkJ|Oq~JkvNO^^z{oi^26$0P~S6Nxz^<;LlLz'
    'ik|5ts{P~wg3HfFf^y;)`03wRo6L>I{=mPswiWQ`>lXZ<ukAh2@Vd5_qW!n}W}Fy`)W~TNEM5^;7T6#Y18_84E-_LM'
    '7MaqC@WYHj8Wm>44<ViNeT)Za)0EF(xhfg4UK%EXreaW<Of{=nU!6lT!6BU^Tn=_hDX)1gd`jy85U=~2z*`;E04^~)'
    'DSl@3vGSl^K;%+F>rmt}n}mrecd86HN3jb;E!EbpuScPQKaBS5r7J?MRC9Ev8=bAEtK49@^4~yf@ET#UdA{iWvX4+U'
    'JpY1PQR1&`YD-p>kCXJCQUfZ}^+SPSqD|L@iF>z#9|5S4iZdE;*68qA*~FVXp|S-CXQMlT`=C*!DAH5qO}hz8?@al('
    '?Ap8%6%BeB<(-JSVDSfj;*t@kwi&o{tHh)AsrgN}EzoQGy4hw52{Sk*`_sJzNGp1M;~ErFrNH(l@iNzxAkUHzOdcL;'
    '9AG{Z8CvukWFmUq`lPsCE(hZDT6Gk_KX0_2BfM~<$=6WYvJ~Fun~o6x5OwfEqV07?a<RPtd;QE&pCQ|R>-9?q3mUzC'
    '%Ab&PZ`l%fq?Aw)*lh{E#i0h==dAuY>FB!0M5A*dtwSMeB|A4Xg>91HvE&6z-n)3)F73X$gIiII$fn9izVdeG@)e>G'
    'OqHMH6L=XTwjzm1a+zm7_Gi3weUX@Hk$jZt7%js8pv3z34Nzdop2I#&Q*zzJTW#X0<U926)`__AD3S<3QgM>SWly&?'
    '$shQSRMm#1D1bqZ7>TF?n!A~Kq^c7eFy)O?x;C$De?VFVo^T+)LEZ_q@A4r*s|H+o=v!LiUpCwX)liVkc;9N>AdmE6'
    '&M(Bgg7|Nf_7-Q(G2(dL`IAoAk2TBa7kab-!z@6W6Nl}fTNjkFK{itxy_XGI-I`^C%VpY*_OGWGJmU{F%b}7|ZW{^e'
    '0r!Z&;sxodig+xonR5F$xL38nuWH8S^Q0o_A|9$m+C?u@lE}|+_ZB~Ycs!gANJe{@aY!F19VO6PuJuZ<z&CoOG$SBr'
    '<{WXsj(V&CyaHhO7*Mj#+7GhaJxHzd({m5rI`OdI&wKsr>QK1Xadn$hhMxNK|EM|?&=>7@q<yR3+5GjlT~XJLFVyuu'
    '6a7rzx=+^HL~@8r^)+v5V*xg<oVOj7=)cqFCpAmwQJDH4cItYso69U%>pnPz06)bXk2YVf887o(Fl<i=-P0t)lzVHp'
    '8jl<iycpT*i%P;#>k%pC%)HZ*%{@MO&x?%b7#^Kf+0Vr--ue{9piPRMA@-5flbvx#p=K3+-n2^KCEyFnW+85%35mB9'
    'ZTC7T^%Qm~po_^~_bdN(o<dl%=9|;7mjo-kCK^^@!o&tP^{v7jL|@8P{<bcNaM6H1e1Go;d9*34uuyvCpiP0%nyQvf'
    'DfxJ(33zs$5M-TBom`+c6S-bgI9aMhEo_i>%3Nk;YYVaAuJfxhtGxeuLg-Fwy;u%5oy0;^J55xr{cddTT6=v$-4n5w'
    'lj0i2{TMK$wVhH>GCJ&gYP(5dUaWXmrq=(hs{0W_yd)1@UMHD-Zf9=++|M9~>*r;I`dYp5RoS3xD|ubB)=f3erppy0'
    'F0Pspn|{%)l^3=jfh_n<zh2@}0IL;6Z)8>Pwn%i~))Q*&d;_x1u}=fYDockLr~DbnMsy|=VsEIy)0KYtT+HNFfoyxa'
    ';`L)}c-4%)97GWjC6hD_HX;lib@`bECeJ@`(|*dA-Fn^gllRdm*9M7dX)$YcfPmh*5U|Z>HWs~SlV)n2PWU*%f-{6f'
    'c8x^roR60fQy@R;n(d3Jz3%d!T30H$f#f>v>yM%TR<@~2!akUyQp{2}%KE3Q9Dhc`9%3@@4pIEhWQTd4I7|w~DdD_;'
    'waX1s9q1xl`tZ#WSoB}zZCM9w+GgI<6}Ww|4Hpa_8A{x3c40k9Pgg)ZyMvy}Re|h(P6F|c%p7vn^TJiKKCKvFdP@>|'
    '-JC(_;&Kyr7ep7T)`qT@uV9+l{BII!lyn*r8of#ATSDqCp;Y7$gR~gbtUI}FKey|+RLgWM_ffQlVlQ}#cI3QPMLmFr'
    '#-JV?AC&`V%gb|jodXwR9N4!5`2(VGRF&wX7*D_u0I|`4ivt*L%p%oCk%wB4l?5)qG;1bUH*>B&ysCDY+1|?A?~itc'
    '7r!PK_CShXHV^n!e}>DrH3tYV0=btQtHQ$R-(R|d<usvD%@QC@YN{p3idp=LZFI2()0_Yz>`%ioUGeMKS}k>U1YwBX'
    'q12}grFiFQBvXJBqeR`x>$v3-xSJBuA_WVqn$$PIDkmuZ^sCYjXZZMC0jt%zRy#>hnj+!0KMCQSR!Ywpp4s!}Wkios'
    'dnqG14=Pn%sL4)Dm5bLTpPYo9tsY8<7<1Q}szDBV77ZmD06vjX^3nLOil`>T7;gM%FFU=21cQP(0Wmw5Chg|u*iWC8'
    '7jrk8igul4OVPAIg#-75wi3gPO|4GB;<~{6_DRl#qN%--u`l;7M%5>-EWRH>0&6WDT?V?tVF8;{R+xG>epRyG?jnN-'
    ';BO$U%;I`Mpz9NJ<R+w8%7x3R4yx3a@JKTMR8e4wN<u_-De2ehq#C_0P`o$o8({RP^Bvz)JAhB8N12^vn`G+*(AmGV'
    '0om-3|KSES|J@Dvqwy-QT_3UA1{8D0m?<j=0m<`|T@XE)Bs>c-Hoq^{6V_1lq83@`&8W-`!Z+^^R)i~1YGrhHxvb7w'
    'gy(nn<;({`(T;I`w1nDm|K=*tCq1Gx8Ozj{?Cd0XlwwJ-Fs<AwLENy&Ft2Q0qS|x96nT}M9haogG1InxTg(>K#;o~K'
    '%!cKYT>_+=W=f}u-8w%#ikVm%-^9!+Qp39RGg&)V;F}Zd?BejTiORB|w5!Z=jv(*bPZ!AJBZy4&`;9``Nqht)!X=uE'
    ';ekeu>Cq=nnA=9UTjB>@4D<c%+=&bHnYknu<;p$Z|6YJY71hEK%$;Kk(TsyF3iRo|80CFJV3!7>EIF_8_q&NHBzN-^'
    '#8sy^TkNm74d3UA9DF1gv!6YMn|R~V#GKJp{gWcZC6+66dB$O}9RKBbyY6Qk7OU7`i=4%)u-%=rU{7p6q}2X)Kf6+-'
    '^Q_BvU@kBFP)vHw(P7Y2xNyOIJJgbO3HrP6nSH059dYUTd^{JsxXH@<d=b$PX|^f#L@91?Qm5TL9c))7YqAPm?})sH'
    '4!Q>cn~74^UQ48r5k}?Z;Ou8(lG|B7-KPRDbb5SIDm&#KFELA#sUUPui>l<ta3}0ZY(mu>Or)fg{hDe133w_!XPO0w'
    '4(;OJ;+#sgxiuB*ajjlFYF`h4@_3wYp-R?Ey!b5(^bmP4If@@64`{~4%_^MKXkwPHtGj_B4;68&5qa=@9eL<Gb*agR'
    '$b(vgaGlW*UOwBF_%(YMmQ<}#nAq;&2XKzP4cIB@{!t@Hxng;X;N4?%P}XDAPWn))1|4T8T55*iIRS~Q!!I5Jv0XA#'
    'zPZq)Z+&K*5c-W|;rTgD7w=)~|N5Imn;uGk-t=>;Oux<6fgYZm&*+~w-POp_hB*3R@QI-IDAv&2q%n#$injm1cU`)X'
    '>VNH*fUh^5l<6y|<W|4P%$`lCB<Tl(9B_%-7@I%fj@%|B`cTUR(>QhfNIe5vv&u@mQ5yfEuOshXirUF#*I3C~ERVwi'
    'h6)p2%XKz$`YiF5^c;N5<S+W07ro?l```$>AP%2Pp-CYCn^XV-yHqO7!ZmZmt=_<i?}wKOur{*z@(I*zC+SN;7(%{y'
    'VK-q8=%-avjOGftvPG8HY`=Sft<pbF8k5JCzMc_Vb0e(tfat@%K+nT}AU?ySkEv|akZ@)IH*5v8NBuq1daBD<O0b8W'
    'ODGZLA$M7dMhm@R<<vflayYy=t;Q*_2@7(!xJxuT4WzndrddYm<uNT8aAGHE58x1Gg%|;#JR#&}lYsL~nG;WvH#I<0'
    '0#gv=(v+xM<={CApu!2*76{vEmQ{A*Eeu7fFSX;QbzuDSjQ=R1S2F%rHX(zER(Vi$!Q*7GO{cY*E&PM&D)>Kc!@+Ey'
    'hkD>*y_0+k3;=QU#81TKVK`WEE$2~wAMFxV!%W$l!B@}XiYcZFyjBSXiJ5lqB{{be3V<-$Z&Kc5<cHRuc%xPaKap`E'
    'p)cykW!S&>Hc^HRn@SO>c;+T*-4F^~D7N<}50`3ooy|VpYGNh4b#{|KFY);CaIl|`EOTt5p1_^sw_SH=>^j`;+a$Oz'
    '28OXGpFuNGT=Y1D51U;m;zFiwUUrm`#K`8;$D!-p=411(Z9cTKwddZ3UU4!mA2sGoLp*98D+btZuV!E75_7>_v4q(0'
    '<I{t0`)0L?3;1&AANU(}EzJ8^Ib|RDkgE%P^e8M2-)FUVSFeMo5se>@{-B=gc>r>5FdcQ;u=MHb6ZpT_x_u&Oc*|x%'
    '^Zn*>@Oi<|;&@!CZAyz?t%F(DM_l!{gNZB@s%?79QQq-SK*c}cx8a|u*NEb6{I+w{t9Q77v!E|^SIuiPN^>{F-H!Ep'
    'fY#|H4o3U%IqdyXjK3Uy*RgdeX4Q@!l--aZ(0U&4IU1juG}S2UCASHf>d>f~1u%b4!TB<=tN%{A!JkDh=dsn(f8HC_'
    '$~NcRmtk3}8kSX=_!^ca4yKB+UD=vfD$wW<9HJCDwXthn>yHEvjt6&);$MyJS_P5R=M#7v!qZJgb8HvFUarQ>__I>q'
    'yzBZKO(ok?K|1T2F*7&5sCA$Rj;!{C_gix&PO={RX;ng$Ygc{Y`kaOfe0|(gdDk(4rNwNw_bZba;qknw*5K!PGI+=>'
    'qIoif*LuVsW)w3PyJL&C9u`<=LvHiPr~$c^k^6>!)~e!JaQ9?8wZ|hi0CZ|0N-!F;%J>Wpj~uTRhlA{QttzC@DrfQ?'
    '<I|Hbhgox0y+SVUwA37EqL4NMM?o>*At<E+^|)<ubFnh0xavAf1twpr%MVipaY9On1MtnCiF7ndANm<aCKN3bmu$_0'
    'd&DWKHJGc`D<b5cHY$^Z<M)za7u;2~f*%bC`sQ)X#VQj7)8`!QT_}F3SPrT=^)k)3d>-7F#?ilKgZk=1(osE~Xu8<Z'
    'kX-y1#nTa;sra=cRjx&idN?{agtpO3=e)INkltZuRA2_1TNX-p5#rGAr(Y;1_LfIpf00MV!^Y1o+UiIb=Gbw{7bO=1'
    '!%AW8JhHq#f00KfFRPD&F4Y%{1iC{)dnP<=@<Xohg9eGN43D!^hk?W&9!^JOpiV7)c--aQ>v$X{0+r#dFzx(>vesMN'
    'pM#cgoCrFYu!&8@hcJQp?`!PGsmQ^?XL<wS75lM({JY6f^8R4#)Zug+q$TqDkp{>z2^%g@eLGXbnp5?DM(6G)kabq)'
    'qS#j`6<*ey45m!@of$E;G3s>j-Qp&@Ywxe|lgDBAUIzm|JYFuY$$wEtQtlPif?0&LGVR$j((+sjqC?HrAsGoiR0~Dw'
    'LAKR!SSz=wu)PmN0(Dcc^Dlz}Ux*|>q<E}r!LstEfDdCTD%Sx(j8JG7f@@p@)7Li17^ONqKd%O(9{LJmwOIRK(~9R2'
    'jHF|~*NSLPYpD1QVD{+EutDCuQnu0IA!|>+F)Fe7Ju^c6J7r7p5>h$z`(dcH>7YAXOd^Bn6yZR%1;F_Es9I?mP9QW>'
    'uWsukJu<=TN_24i#~!$pRwuxj=S036KSR$9p3U4Up>;M(`~LKO;`QxpCewP;r(WfPayFwo0ZodQUrj9Z$e_X_Rf60j'
    '%U$+i>#~jFvCL8O96LprSW8^r=gGnj^DrAW8uYi3cbX4r>g?HGo_i<_LDvxhoSyg+D$EtA;Ew8Su+;tZ>VxoY-Un2e'
    'v8;lXpCFb@PQdCzFFSH%n*Ry~7Ea%dWPeLRBq+zfq(D|FaICG_*W1Z(G@~NF1lw~)O7zElTZAXILSNtfHqtOxMk(fe'
    '+Zv!n!{=I=br$GP^1tALdUb$e$!`ZvalCA`&j+r}dfFd@4UeMjJO+aO?Hryp7x3sKeM~FZkaRZr#pfvC|C<Z&UpyCp'
    'xYsF*V@6z5`r@L~2B1mqELMZg@+uO(x3bZ@NK6Q;0CynsA`+gqyB9=lKeWMW?*Kc|2(j5t*(g#YSDFiyt0t-7(4%yh'
    '9f}>J38CBJmh^7hZZ}D2*J~-<9u?{=gq8X!;x>KM=N*rOJDmBRU@BEsKn0jqqA6RCi6)ahBH7z?_~tc6-*y0Y(1TC7'
    'Q4=SO`qRLt+@*P2zTE(CrU5KuLfT0v_&D^V73a%yFU5NsAMf_Q#zw_Fy~30JIILv1wA5y^#?16u@KEkM##g(+gvDqC'
    '%K&k&x#0^Z(G;PhT!?Ze)jW0g;uJcDAg+FbBK&G_98Sak_904oPlN_Iz#L-puH-><#Z&r+J3}6AorJ1WQJtq&e&R_L'
    'VZlGClYg4Z12YtuoF!hbFL@z_NXK~qx3(vA0bq0}MOtsphS#L!Y}_j(-g2ri2X##sz_^)1iYmSRIl=(=YCnXLAbKth'
    '{0aR}&OOnzsl@Z;Z9;z=MNmf3?C&=L{}9c7pUbQXe;s)@B9dN;dF)SaGx>hlfcm&4lJ6UE!+h@=M@?3QKkvIh3+U#_'
    '=xdb%Y>I{%M;P&5tOSfJ#sIZ+6weCbNYl=94Yb=ZELeQs?V|h3-TrS{m;ABGGiUD~yO1i28u;-Apqe$uHD*{fxIoM#'
    '3*Cx0d%tz#w^#f5w%5(Ab|&&zTy^eVxYEFQJKb2|^DdVfiF5XKsyTGSzwa~6)J-R2eA?}<lnhm7NyOeWRF<7)h6<fa'
    'rC#pXTz#ncPH-s6$L}TCrIc1ar`oR}fi&u=nmm>&z!v#KQWPG>CBi7)56Zri-ug80SsG~iA=vd4ojJZ=C3hQ0*bhsU'
    '&(RrsW1d%{GtR0n?B*&`0ZJWq?XP=2LY-w#;|M9};~#cWXPl*E0UxqSejJ3iANT=U-_dB68I$H(M|4xz^}<;jivV6c'
    '(Ieqaw1YDI292%>9KT&Htj$t*z}|<j-HVe9IV1*Z?;D|;YKZ}CB{V}Z5H-Z0b%z%p_Z^QpDz7&pyb$?5b5FAY?&ZRO'
    '*9#kgQ&S2Fosf^Z6~7f=MMw=^*qBkL1HuSRaB3B^Kb}X1QV44F3Cn8ToU3h^wi#rYm|kvp1Ya{7K696>t+r>(V@kFy'
    '_qsgh*v>~#z0GY#P8Yrz8LOW(1Kb92q5%pPt$FFMk&>vTi)7cMmP1B$mM7Vj)FFElZ+zDxTRXlmOL#1zDbOrl$(^SF'
    'h747e_rclsDE_Y8+IY(k{AP@axvl$VjL9T#Nh&=osQ7%yL@bXk37wfghIqJ4fW<DUBAle;z2AlddzfhSK3GVbhpQ~z'
    'uKGlssK01i&@KynoGgf|^88^)w4D)>=KRtpZz$Fg$0RW|DBMmS3#s=4L~_Y9<!#!1AB6Y|KL2^?*z2@LQ=s2s_j?N$'
    'F^DrD%B!|AAsFb5fpSCB?Q$kmC;o>-*7@6&L$mRII-3mOyjsOP_}_vNcPp&(t;vrLKhP5KrqDfc6jN?Ba_iunuH%Xq'
    'oGELS{g}vJdzSiKUGN`yCC2;R@=yKIC65BHk9r0UL7pU^wV1uO^^AxA72_<jUyidlsz)blMuTkaFYI&>WOBy>Br#B~'
    'LjEtg&Eb!ys(WfhGySZchJrO>%1Layfn5X8zbw1Cg~61Xo5trle4-fkWk06~tEGyJs+e~Af*-m1uvnqaYQ;B*bYE`0'
    '$2W4&l|%6j*7PyH0gwsI|G<F;kU53OaqWL(4!kHC*<F>ON#Oy2z|K>kVR|^a99scWi{9ak25U*4^&Z|8=d#@g<7sj*'
    '+8JEH?=F0gPu{C))v>STZ_*wY(-}4>8S0v854^pe^1k9uhC7|5_<bR*7Fhi%^b7BpE`J&N)vFE@-9Kq%%9Y@=VcXck'
    '`Cb`j8Ra8r%-HV+5Kq3H*467N{cXYNIim!IU+a34F)CMW;s!SB2p@Bo$!Y~FPh*r|JlEhg%1$_*Fa|A%Tj*@L0Du^&'
    '6dXL{MT4GIo|+R!gzSD{|B}Na_Vp%ru->Qu7E=@zVKvPH!RJwMV_^UECQ0ws?Kq2`sKek?ksBT9xER@Uag|6zkboc&'
    '=K@L7SyAi!gePI$v9d~rjxNIZAE%+<R=icizD?C83FZF6eSe|XxX8*?+qfzKjn+!KhnfHXV3u|_f=zmwk-%$~Rh>ZY'
    '&~?PuNFWl-e6bH;TNmuM${bf1(!%_r(3tnJ6WAtOMNh+jJon@sElzu#m(<-)#<Xenm6toC_M9D+yghWG10LtHZtrrE'
    'sG;}`yq|RN9>2NJU&e2SU-UB>sd4r`HrBm$`=59TFX%>ox?#3P1jR?R&vG63LpmgPyX)qyHl9*%Zvq>x-b8U5>)beX'
    '!U7f<$sdF;O4a+_<8vJUjx%5m#j#3#m$Jo6Xn4=$W3kbF?mTWkpDS>ROj1p_p!aEBz4x9piis-`T}7;p{qcR?XPe7&'
    '{A?=E?qsWMDUrjiHJ3*q3&yAd9?evMVeOB9)mSp@eKeJvzMHduM&EZA6%yn&FBaqL?ov*Pua(KiyL~?yL(kBV*DoWN'
    'r=e|kV`wzk@KmuC`UG$}1>gc<oT&9$8aLjY8`~^!0?Grlwx@J4j3yPloBIso+ZbFsyi+K_1uY!-PsLZ|f?HEV5`w9m'
    'GM@FrqhkP|{#hPgQeCF$SI9S{)LyD%5eVD-ZL)3Mf03MymCWU`R2}FhF^x33fHh)bi!t^b++@tw9$TXj3?+v@==t?N'
    'mgN9dI-VrAx^F`y@a`~#T689OjgDD(cPY0R8pjf;>dT>f;6a1jOZ0*li;}z7;#3j+VF&_--maN5;_}qk1y9~zf4dcg'
    'YWs{(8QR_aeCz-WbTM?bzAEi4V^rk`E3_JJO@s=-ZtN?B`X4t81#a_Oj{6}L8F#X3undJF-S+KsIL8alSvv%aJRNYG'
    '&v4HEUNjc;KJ}^!tis-2y?k44g%e^TJQThp_bt`*hD`3MCRm4$2TLy?@p@%-Y>03!k~8!}pQw6xxO<U=?<zEnN?OlH'
    'D`-e5f_IwbU{Aj0OW3jWo`I0|qG_4HxL{1e(EtyiyZG$2eTpEV4$iXjG(1-3hsD<9_bNb{ya^mmJ&VSlwtf_Lgne_6'
    'lRQWDfb&M|CNEd3*VaUq5HMMYA!`9!!+6Av_cZXIUDKGnR*W@U63=>l@Hdbr|8}3yt=X;5ks^CX)v3+UzkaF`?ayHZ'
    '&Y9cftUCcqMBS2DpquD9O0Zk1tSoF!ygrnru9rK_@7+j0Y1MNNahbTc>OC%@wnAs{*0g6C{cHn@BH91|Q>!tJ0e$J('
    'x{?et^Od{zaABz24F`lyr|;z^<fQ;(j~?Dfn*v~MYcIOE08Q6HIQ$Po*Q~Gr7{zJ7E8^H-kzREsx9Zo^7uByzo$Cwy'
    'qnPuO4JWG&RO*N_n`uq_W6ySrAIto%q%o+^4*@jr-_F%moV=fHg&fwW;jczstLTEmm6<kXO*#Ra<z`oz<ZTCwNTE4~'
    '&}tm>npR`5YE<5dz28oS%|2Jj&<+oEzsNgv@tq#8(Ipn=)*Y;sv0(GU_Ph=8HmGqAwUiyb02E%d&Kx7thT?!}JrezC'
    'AC5?PYP`O~Hkd$+cxu`=#h@t^uQmSJZvHv_n}TESGA(XpxBOp;V6s|6$!|KW?HlED%8Ws+>Ih=UEMQD?3IS^hwV|M='
    'f9$}LsiWgQ8~a_Jg+W?h9G-b>O)X(tW&b{zI!o76z4zfgS0DW9H+nyo$fEIwY*4iLL}RS6R%59@zhj`^B3lW7H>M{z'
    '3v+{^bv(%ZPzVelb3{oO2sD(r5L<t4f-9j9dlu*vr)T?V7)DZ!sSlR)TaX_|Q*P!{i3J<;JL+5CQ(pyVb+kt>#HP3{'
    'qwvqAKlJd3Fv)!r9p;j)Jy_+`uS_l9ucoEv>;Gnw?wks6fdQq=%Klx5&iQSz=INrug`5H5pix7UsBPv<q~e}<*4$(='
    'ABd)jgf!vruBS_L-vk1@d&%=qkm49r_$R^KR+*E>nt^%J06S}${SsqkcI=!;%HUNQB7mO&7B4O2D?Kj0m`$a^<;6=;'
    '&5vRsLvJj+Tga1>i+_v+-B9NO9sdVq4Trg#u_DuXiv$&$MU6<1QginoTZZr-Hh+*c4Y1_LQB+0!I18LAd*AkHMC%0D'
    'AORJ!qZvy-nep}5`Z7<Hp;pU(VYGb#)K<P$jrQ7DevO|tj)5IJ4cowRU8Hya#By(juCwRMO!iPQ+KEmMHFJkwgFk0{'
    'q*pExxx`0==oafEE*SD%D89~Tqvv*rhRMz_02)0vcrF!@#5z8Zc&_L~8E=vDa+o^ls@aEA)r>i&v~v5nXldbP^TnwZ'
    'yf2?suABXJo1I!m430gnocqvc*hwx0{39~Hfrs8oWadZ|;q>cJR1+DWvvCc2Z~d>@B^(Oa?8N;HOtQFYO8tXxo5i&H'
    '><~f4VQWT9I#0-S&o2w{<q2oQ#;p#PWG9|S;ga|J`!v}{w%i+~s_NnKj6MIjgd-a~)J@M9L9)Q$cq;Kpu8BssC1CIU'
    '2|Y*gay27Z@SYEiZ&4E}*f=j56Var4X_k95`DL<gVV`#!UkW~>2!8CGpJ_%tD@;b$wr4MXQP<})#Yu!ZgD;35QEd!h'
    '#3A*6ctvkB_Fh-Ss-pwl6*;R69sjr@SC}dkQilO04)Rf0dFg`GmxQ?Jr_imD%3a+XizO|2#uKM9Kt^niSgt+CqEGjs'
    'mVP~tHQFsOKez`INR-=z+T|3q<sz8E^QFAZwfUpj|C6}KZH8B1Aawl+>hZS97kBo5{!gS~(37`b&dO(O5p3bmo@+PX'
    'fA0zquF)n2@3vH$@2i^1#}uv#U!*188^y2i8&>8*`0c?<jSpQ;6+MGZdhtKFK&?Mkr^A_pLj+*&-i&?4GQPdO;7EV('
    'M9PW~C%qfk{sgCHJPYw^Yu*BLQcobgSjd?7HDGL<Czx4I0%&Z-1+3xb6~n^s!WRKcq~p-foKalbZdHdO!6fy4cvo+|'
    '_bAWt;W`lZ96`dNNTZ*}Yp(78h$_?=4+o|@zF)7X9@_qVrKN_5gQ>J<%H232^tgIV=m4P6V%?J%TyG5agyew-HcPd*'
    'm{0}l7zAO&k^+avQibpH!c@G6O~!+L7~zgE<(?tHCe6OXA2j{n8fW1Ec0+SJfR%^*4^D2S=;SeQB15N7CwD3u^zHB;'
    '_qe7|sSh_An*;15y<0O!`|23$-Dg|&VOF@h5L=_p?Y!1=pD{r)Jc7}=4;R*~d*K$BS(7?oc-;nZDHQDi!rmDpLrFwJ'
    '>CJnUcQ~ziQ%F6$bp40jdhjv)qeM}L87c$w$Sm69c+|cdKJW1uMNwFJ1{-44F{l~%+)<fmiDsX1R5cGr&>m!wIDS{`'
    '<;Xe~f??L$pkcmnGOX#L7)_m&z-~3O?@!ZqbIR$9hQ}Sa`}AALia+-yHfRWWjE0j%89TyTd?Xq!$_2``7%WpxbEEV2'
    '?lb&**E&Fl_<dHxEsAojOiYPIP3pJ0xgO{a+l0j`lQkQz;dMoUAe1O~FBEja7_=4mAC1Xdd4w>U1a`INX4rRzie8F7'
    't<>C&=D138xrYW{G1?qAZ4Z=lvpE15!Vth&G*jrlqiZ8&z^=^PG|FD;j|l%b!VHo|f1B$v<djC{59iWbTZN1}po^#}'
    'uU{YG*7hzBmDapm$+Z}x9HE4kltfj*@MFKeuQ0xoJ5zYcOHmpEWD*ry{m!j(@-26B2Pd=XU<a9>d{K5N-Ik|)0G#x='
    'VSycQOamyMeWo-#AMg|@UcxQk$Of5Up6^R+{VfC%PaQNuAW97Ezi~6ud(_~yHgBP}W#=qD_$Tpx);hb^e$!k6M=oXg'
    '#e7YGc#Iz|nVk%f9#->Gu)yr4A7OpU_|d9<9GiApc23b1^IVk6V`GyeE*LCfI5BkUYLQZVt@LuO3|$(AOPSU_pG^an'
    'U~zR9PDx}`lq4HW1O4y+4<A;ST>'
)


@cache
def descriptions() -> dict[tuple[str, int], bytes]:
    """Self-contained ROOT class descriptions may be carried into any output file."""
    rows = json.loads(zlib.decompress(base64.b85decode(ENTRIES)))
    return {(name, version): base64.b85decode(raw) for name, version, raw in rows}
