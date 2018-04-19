# uvoz potrebnih modulov
from gonilo import Gonilo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

# kreacija objekta
Klemen = Gonilo(7500, 1455, 100, 2.5, 4.5, z1=21, z3=21, psim=20)
Klemen.psim = 20

# statika gredi
Klemen.statika_gredi_1(l1 = 63, l2 = 78 + 43)
print('STATIKA GREDI 1')
print(f'Obremenitev ležaja v podpori A: {Klemen.gred1_A:0.2f} N')
print(f'Obremenitev ležaja v podpori B: {Klemen.gred1_B:0.2f} N')
print(f'Maksimalni moment na gredi 1: {Klemen.gred1_Mmax:0.2f} Nm')
Klemen.statika_gredi_2(l1 = 63, l2 = 78, l3 = 43)
Klemen.statika_gredi_3(l1 = 63 + 78, l2 = 43)


# laboratorijska vaja
Klemen.LV_preberi_meritve('podatki_laboratorijska\S01meh.txt')
print('\n\n')
print('LABORATORISJKA VAJA')
print(f'Delo trenja: {Klemen.LV_Wtr:.1f} J')
print(f'Realna razlika temperatur sklopke: {Klemen.LV_TsklR(22.8, 28.2):.2f} °C')
print(f'Teoretična razlika temperatur sklopke: {Klemen.LV_TsklT(8.3, 460):.2f} °C')
print(f'Maksimalno dovoljeno št. zagonov ob upoštevanju realne razlike T: {Klemen.LV_nzagonov(373)[0]:.0f}')
print(f'Maksimalno dovoljeno št. zagonov ob upoštevanju teoretične razlike T: {Klemen.LV_nzagonov(373)[1]:.0f}')
print('\n\n')

# statika ležajev
print('STATIKA LEŽAJEV')
Klemen.lezaji_staticna(gred = 1, fiksen = 'A' , C0={'A':8300, 'B':8300})
Klemen.lezaji_staticna(gred = 2, fiksen = 'A' , C0={'A':14600, 'B':16000})
Klemen.lezaji_staticna(gred = 3, fiksen = 'A' , C0={'A':23200, 'B':23200})

Klemen.lezaji_doba_trajana(gred = 1, fiksen='A', C = {'A':13800, 'B':13800})
Klemen.lezaji_doba_trajana(gred = 2, fiksen='A', C = {'A':22100, 'B':22900})
Klemen.lezaji_doba_trajana(gred = 3, fiksen='A', C = {'A':30700, 'B':30700})

Klemen.lezaji_razsirjena_doba_trajana(gred = 1, T_obr = 80, Cu = {'A':355, 'B':355}, d_lez = {'A':30, 'B':30}, D_lez = {'A':55, 'B':55}, olje = 100)
Klemen.lezaji_razsirjena_doba_trajana(gred = 2, T_obr = 80, Cu = {'A':640, 'B':710}, d_lez = {'A':45, 'B':50}, D_lez = {'A':75, 'B':80}, olje = 1500)
Klemen.lezaji_razsirjena_doba_trajana(gred = 3, T_obr = 80, Cu = {'A':980, 'B':980}, d_lez = {'A':60, 'B':60}, D_lez = {'A':95, 'B':95}, olje = 1500)

# kritični prerezi na gredeh
# če podamo tabela, se računa prerez pod zobnikom, če pa podamo graf, se računa stopnica
Klemen.kriticni_prerezi(gred=1, Rp02=215, Rm=500, Rmax=4, material='S235', tabela = {'stevilka': 5, 'd_pod_zobnikom': 35})
Klemen.kriticni_prerezi(gred=1, Rp02=215, Rm=500, Rmax=4, material='S235', graf = {'d':35, 'D':45, 'x_od_roba':120})

Klemen.kriticni_prerezi(gred=2, Rp02=215, Rm=500, Rmax=4, material='S235', tabela = {'stevilka': 5, 'd_pod_zobnikom': 35})
Klemen.kriticni_prerezi(gred=2, Rp02=215, Rm=500, Rmax=4, material='S235', graf = {'d':50, 'D':55, 'x_od_roba':120})

Klemen.kriticni_prerezi(gred=3, Rp02=300, Rm=600, Rmax=4, material='S235', tabela = {'stevilka': 5, 'd_pod_zobnikom': 35})
Klemen.kriticni_prerezi(gred=3, Rp02=215, Rm=500, Rmax=4, material='S235', graf = {'d':65, 'D':75, 'x_od_roba':120})

# povesi in zasuki gredi
Klemen.povesi_zasuki(gred=1, d=40, E=210000, previs = {'F':1000, 'a':30})
Klemen.povesi_zasuki(gred=2, d=55, E=210000)
Klemen.povesi_zasuki(gred=3, d=70, E=210000, previs = {'F':1000, 'a':30}) # lahko vpišemo previs ali pa ne. F je sila, a pa oddaljenost sile od ležaja


print('\nGred 1:')
print('Statika:',pd.DataFrame(Klemen.gredi)[1]['statika'])
print('Kriticni prerezi pod zobniki:',pd.DataFrame(Klemen.gredi)[1]['prerezi']['pod zobnikom'])
print('Kriticni prerezi stopnica:',pd.DataFrame(Klemen.gredi)[1]['prerezi']['stopnica'])
print('Povesi:',pd.DataFrame(Klemen.gredi)[1]['povesi'])
print('Zasuki:',pd.DataFrame(Klemen.gredi)[1]['zasuki'])

print('\nGred 2:')
print('Statika:',pd.DataFrame(Klemen.gredi)[2]['statika'])
print('Kriticni prerezi pod zobniki:',pd.DataFrame(Klemen.gredi)[2]['prerezi']['pod zobnikom'])
print('Kriticni prerezi stopnica:',pd.DataFrame(Klemen.gredi)[2]['prerezi']['stopnica'])
print('Povesi:',pd.DataFrame(Klemen.gredi)[2]['povesi'])
print('Zasuki:',pd.DataFrame(Klemen.gredi)[2]['zasuki'])

print('\nGred 3:')
print('Statika:',pd.DataFrame(Klemen.gredi)[3]['statika'])
print('Kriticni prerezi pod zobniki:',pd.DataFrame(Klemen.gredi)[3]['prerezi']['pod zobnikom'])
print('Kriticni prerezi stopnica:',pd.DataFrame(Klemen.gredi)[3]['prerezi']['stopnica'])
print('Povesi:',pd.DataFrame(Klemen.gredi)[3]['povesi'])
print('Zasuki:',pd.DataFrame(Klemen.gredi)[3]['zasuki'])

# korenska in bocna nosilnosti
Klemen.korenska_nosilnost(zveza=12, Yfa=1, Ysa=1, Kfa=1, Kv=1, sigmaflim=250, YRrelT=1, Yx=1)
Klemen.korenska_nosilnost(zveza=34, Yfa=1, Ysa=1, Kfa=1, Kv=1, sigmaflim=250, YRrelT=1, Yx=1)

Klemen.bocna_nosilnost(zveza=12, Zeps=0.75, Kv = 1, sigmahlim = 600, ZX=1)
Klemen.bocna_nosilnost(zveza=34, Zeps=0.75, Kv = 1, sigmahlim = 600, ZX=1)

# zapis končnih poročil,
Klemen.zapisi_txt('Porocilo.txt')
