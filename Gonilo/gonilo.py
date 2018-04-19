import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Gonilo:
    """Class za preračune vseh potrebnih parametrov za poročilo pri predmetu Strojni elementi 2.
    Zahtevani vhodni podatki so (Vhodna moč P1[W], vrtilna hitrost prve gredi n1[min^-1], vrtilna hitrost tretje gredi n3[min^-1], modul prvega in 
    drugega zobnika m12[mm], modul tretjega in četrtega zobnika m34[mm], število zob prvega zobnika z1[/] in tretjega zobnika z3[/] ter kota 
    poševnosti beta1[°] in beta2[°].)"""

    # Podatki, ki so značilni za večino instanc, narejenih iz tega razreda.
    # Lahko se tudi spremenijo z naprimer: self.eps1 = vrednost
    epsl = 0.995    #izkoristek ležajev               [/]
    epst = 0.98     #izkoristek tesnil                [/]
    epszd = 0.97    #izkoristek zobniške dvojice      [/]
    taudop = 16     #dopustna strižna napetost gredi  [MPa]

    def __init__(self, P1, n1, n3, m12, m34, z1=17, z3=17, beta1 = 0, beta2 = 0, psim = 15):

        vhodni1 = ['P1', 'n1', 'n3', 'm12', 'm34', 'z1', 'z3', 'beta1', 'beta2']
        vhodni2 = [P1, n1, n3, m12, m34, z1, z3, beta1, beta2]
        for vhod in vhodni2:
            if (type(vhod) != int and type(vhod) != float):
                raise Exception(f'Vhodni podatek {vhodni1[vhodni2.index(vhod)]} mora biti tipa int ali float!')

        self.P1 = P1        #vhodna moč
        self.n1 = n1        #vrtljaji gredi 1
        self.n3 = n3        #vrtljaji gredi 3
        self.m12 = m12      #modul 1. in 2. zobnika
        self.m34 = m34      #modul 3. in 4. zobnika
        self.z1 = z1        #število zob 1. zobnika
        self.z3 = z3        #število zob 3. zobnika
        self.beta1 = beta1  #kot poševnosti 1. in 2. zobnika
        self.beta2 = beta2  #kot poševnosti 3. in 4. zobnika
        self.psim = psim    #modul širine zobnika

        self.preracuni()    #inicializacija metode za preračune

        self.zobniki = {
            1: {"modul": self.m12, "st.zob": self.z1, "d": self.di()[1], "da": self.dai()[1], "df": self.dfi()[1]},
            2: {"modul": self.m12, "st.zob": self.z2, "d": self.di()[2], "da": self.dai()[2], "df": self.dfi()[2]},
            3: {"modul": self.m34, "st.zob": self.z3, "d": self.di()[3], "da": self.dai()[3], "df": self.dfi()[3]},
            4: {"modul": self.m34, "st.zob": self.z4, "d": self.di()[4], "da": self.dai()[4], "df": self.dfi()[4]}}
        self.gredi = {1 : {'statika': {}}, 2: {}, 3: {}}
        self.konci_slovar = {"gredi": self.gredi, "zobniki": self.zobniki}

    def preracuni(self):
        """ Metoda za preračune osnovnih podatkov: omega1 = kotna hitrost 1. gredi [rad/s], omega2 = kotna hitrost 2. gredi [rad/s], omega3 = kotna
        hitrost 3. gredi [rad/s], isk = teoretično skupno prestavno razmerje [/], i12 = teoretična prestava med 1. in 2. zobnikom [/], 
        i34 = teoretična prestava med 3. in 4. zobnikom [/], isk_dej = dejansko skupno prestavno razmerje [/], i12_dej = dejanska prestava 
        med 1. in 2. zobnikom [/], i34_dej = dejanska prestava med 3. in 4. zobnikom [/], Pds = moč na delovnem stroju [W], vrtilni momenti na vseh   
        3 gredeh T1, T2, T3 [Nm], medosna razdalja med 1. in 2. gredjo = ad12 [mm] ter 2. in 3. gredjo = ad23 [mm], 
        širina 1. in 2. zobnika = b12 [mm] ter 3. in 4. zobnika = b34 [mm]."""

        # --------Kotna hitrost prve gredi--------
        self.omega1 = 2 * np.pi * self.n1 / 60

        # --------Teoretične prestave--------
        self.isk = self.n1 / self.n3
        self.i12 = 0.7 * self.isk ** 0.7
        self.i34 = self.isk / self.i12

        # --------Dejanske prestave--------
        self.i12_dej = self.z2 / self.z1
        self.i34_dej = self.z4 / self.z3
        self.isk_dej = self.i12_dej * self.i34_dej

        # --------Dejanske kotne hitrosti--------
        self.omega2 = self.omega1 / self.i12_dej
        self.omega3 = self.omega2 / self.i34_dej

        # --------Moč na delovnem stroju--------
        self.Pds = self.P1 * (self.epsl ** 6 * self.epst ** 2 * self.epszd ** 2)

        # --------Vrtilni momenti na gredeh--------
        self.T1 = self.P1 / self.omega1
        self.T2 = self.T1 * self.i12_dej * self.epsl ** 4 * self.epst * self.epszd
        self.T3 = self.T2 * self.i34_dej * self.epsl ** 2 * self.epst * self.epszd

        # --------Geometrija zobnikov--------
        self.ad12 = (self.di()[1] + self.di()[2]) / 2
        self.ad23 = (self.di()[3] + self.di()[4]) / 2
        self.b12 = self.psim * self.m12
        self.b34 = self.psim * self.m34

    # --------Zobi--------
    @property
    def z2(self):
        """Metoda preračuna in vrne celoštevilsko zaokroženo št. zob 2. zobnika."""
        z2 = self.z1 * self.i12
        if z2 - int(z2) < 0.5:
            return int(z2)
        return int(z2) + 1

    @property
    def z4(self):
        """Metoda preračuna in vrne celoštevilsko zaokroženo št. zob 4. zobnika."""
        z4 = self.z3 * self.i34
        if z4 - int(z4) < 0.5:
            return int(z4)
        return int(z4) + 1

    # --------Geometrija zobnikov--------
    def di(self):
        """Metoda vrne slovar kinematičnih premerov zobnikov.
        Primer: di[1] vrne kinematični premer prvega zobnika."""
        return {1: self.m12 * self.z1, 2: self.m12 * self.z2, 3: self.m34 * self.z3, 4: self.m34 * self.z4}

    def dai(self):
        """Metoda vrne slovar temenskih premerov zobnikov.
        Primer: dai[1] vrne temenski premer prvega zobnika."""
        return {1: self.di()[1] + 2 * self.m12, 2: self.di()[2] + 2 * self.m12,
                3: self.di()[3] + 2 * self.m34, 4: self.di()[4] + 2 * self.m34}

    def dfi(self):
        """Metoda vrne slovar korenskih premerov zobnikov.
        Primer: dfi[1] vrne korenski premer prvega zobnika."""
        return {1: self.di()[1] - 2.5 * self.m12, 2: self.di()[2] - 2.5 * self.m12,
                3: self.di()[3] - 2.5 * self.m34, 4: self.di()[4] - 2.5 * self.m34}

    # --------Minimalni premeri gredi--------
    dstand = [10, 12, 15, 17] + [_ for _ in range(20, 115, 5)] + [120]  # Standardni notranji premeri ležajev (KSP:659)

    def d1(self):
        """Metoda vrne minimalni premer prve gredi, zaokroženo na prvi večji standarni premer."""
        import numpy as np
        d1 = (16 * self.T1 / (self.taudop * np.pi)) ** (1 / 3) * 10
        razlike = [d1 - i for i in self.dstand]
        return d1 + abs(max([n for n in razlike if n < 0]))

    def d2(self):
        """Metoda vrne minimalni premer druge gredi, zaokroženo na prvi večji standarni premer."""
        d2 = (16 * self.T2 / (self.taudop * np.pi)) ** (1 / 3) * 10
        razlike = [d2 - i for i in self.dstand]
        return d2 + abs(max([n for n in razlike if n < 0]))

    def d3(self):
        """Metoda vrne minimalni premer tretje gredi, zaokroženo na prvi večji standarni premer."""
        d3 = (16 * self.T3 / (self.taudop * np.pi)) ** (1 / 3) * 10
        razlike = [d3 - i for i in self.dstand]
        return d3 + abs(max([n for n in razlike if n < 0]))

    # --------Statika gredi 1--------
    def statika_gredi_1(self, l1, l2):
        """Metoda sprejme naslednje podatke: razdalja od podpore A do središča 1. zobnika - l1 [mm], razdalja od središča zobnika 1. do  podpore B 
        pa predstavlja seštevek dolžin l2 in l3 [mm] (njujino razmerje nima vpliva, pomembna le skupna dolžina).
        mm].
        
        Izvedejo se preračuni naslednjih veličin: Tangencialna sila - Ft1 [N], radialna sila - Fr1 [N] in če je uporabljen zobnik s poševnim 
        ozobjem pe aksialna sila - Fa1 [N]. Sile na ležaje v podpori A: Ay [N], Az [N] ter v podpori B:  By [N], Bz [N]. Skupna obremenitev ležaja 
        v podpori A - A [N] in v podpori B - B[N]. Največji upogibni moment - Mmax [Nm].
        
        Ob klicanju metode se prav tako izrišejo grafi prečne sile in upogibnega momenta v X-Y in X-Z ravnini."""

        self.gred1_l1 = l1
        self.gred1_l2 = l2
        self.Ft1 = 2 * self.T1 / (self.di()[1] / 1000)
        self.Fr1 = self.Ft1 * np.tan(20 * np.pi / 180)
        if self.beta1 != 0:
            self.Fa1 = self.Ft1 * np.tan(self.beta1 * np.pi / 180)

        # XY ravnina
        # Ay + By = Fr1
        # Fr1 * l1 = By * (l1 + l2 + l3)
        A1 = np.array([[1, 1],
                      [0, (l1 + l2)]])

        b1 = np.array([self.Fr1, self.Fr1 * self.gred1_l1])

        self.gred1_Ay = np.linalg.solve(A1, b1)[0]
        self.gred1_By = np.linalg.solve(A1, b1)[1]

        # XZ ravnina
        # Az + Bz = Ft1
        # Ft1 * l1 = Bz * (l1 + l2 + l3)
        A2 = np.array([[1, 1],
                      [0, (l1 + l2)]])
        b2 = np.array([self.Ft1, self.Ft1 * self.gred1_l1])

        self.gred1_Az = np.linalg.solve(A2, b2)[0]
        self.gred1_Bz = np.linalg.solve(A2, b2)[1]

        self.statika_gredi_1_graf_xy_T()
        self.statika_gredi_1_graf_xy_M()
        self.statika_gredi_1_graf_xz_T()
        self.statika_gredi_1_graf_xz_M()
        plt.tight_layout()
        plt.show()

        self.gred1_A = (self.gred1_Ay ** 2 + self.gred1_Az ** 2) ** (1 / 2)
        self.gred1_B = (self.gred1_By ** 2 + self.gred1_Bz ** 2) ** (1 / 2)
        self.gred1_Mmax = (self.gred1_Mymax ** 2 + self.gred1_Mzmax ** 2) ** (1 / 2)
        self.gredi[1]['statika']['M_max'] = self.gred1_Mmax

        # Potek momentov
        self.gred1_M = (self.gred1_My **2 + self.gred1_Mz **2) ** (1/2)

    def statika_gredi_1_graf_xy_T(self):
        """Metoda za izris poteka prečne sile na gredi 1 v X-Y ravnini."""
        x = np.arange(0, self.gred1_l1+self.gred1_l2, 0.1)
        T1 = [self.gred1_Ay for _ in x[0: np.nonzero(x == self.gred1_l1)[0][0]]]
        T2 = [-self.gred1_By for _ in x[np.nonzero(x == self.gred1_l1)[0][0]:]]
        T = np.concatenate((T1, T2), 0)

        self.gredi[1] = {'statika': {'Ty': {1: max(T1), 2: max(T2)}}}

        plt.subplot(221)
        plt.title('Graf prečne sile Ty prve gredi')
        plt.plot(x, T, label='Prečna sila')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Ty [N]')
        plt.xlim(0, self.gred1_l1 + self.gred1_l2)
        plt.grid()

    def statika_gredi_1_graf_xy_M(self):
        """Metoda za izris poteka upogibnega momenta na gredi 1 v X-Y ravnini."""
        x = np.arange(0, self.gred1_l1+self.gred1_l2, 0.1)
        M1 = [self.gred1_Ay * _ for _ in x[0: np.nonzero(x == self.gred1_l1)[0][0]]]
        M2 = [-self.Fr1 * (_-self.gred1_l1) + self.gred1_Ay * _ for _ in x[np.nonzero(x == self.gred1_l1)[0][0]: ]]
        self.gred1_Mz = np.concatenate((M1, M2), 0) / 1000

        self.gred1_Mzmax = max(abs(self.gred1_Mz))
        self.gredi[1]['statika']['Mz_max'] = self.gred1_Mzmax

        plt.subplot(223)
        plt.title('Graf momenta Mz prve gredi')
        plt.plot(x, self.gred1_Mz, label='Upogibni moment')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Mz [Nm]')
        plt.xlim(0, self.gred1_l1+self.gred1_l2)
        plt.grid()

    def statika_gredi_1_graf_xz_T(self):
        """Metoda za izris poteka prečne sile na gredi 1 v X-Z ravnini."""
        x = np.arange(0, self.gred1_l1+self.gred1_l2, 0.1)
        T1 = [-self.gred1_Az for _ in x[0: np.nonzero(x == self.gred1_l1)[0][0]]]
        T2 = [self.gred1_Bz for _ in x[np.nonzero(x == self.gred1_l1)[0][0]: ]]
        T = np.concatenate((T1, T2), 0)

        self.gredi[1]['statika']['Tz'] = {1: max(T1), 2: max(T2)}

        plt.subplot(222)
        plt.title('Graf prečne sile Tz prve gredi')
        plt.plot(x, T, label='Prečna sila')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Tz [N]')
        plt.xlim(0, self.gred1_l1+self.gred1_l2)
        plt.grid()

    def statika_gredi_1_graf_xz_M(self):
        """Metoda za izris poteka upogibnega momenta na gredi 1 v X-Z ravnini."""
        x = np.arange(0, self.gred1_l1+self.gred1_l2, 0.1)
        M1 = [-self.gred1_Az * _ for _ in x[0: np.nonzero(x == self.gred1_l1)[0][0]]]
        M2 = [self.Ft1 * (_-self.gred1_l1) - self.gred1_Az * _ for _ in x[np.nonzero(x == self.gred1_l1)[0][0]: ]]
        self.gred1_My = np.concatenate((M1, M2), 0) / 1000

        self.gred1_Mymax = max(abs(self.gred1_My))
        self.gredi[1]['statika']['My_max'] = self.gred1_Mymax

        plt.subplot(224)
        plt.title('Graf momenta My prve gredi')
        plt.plot(x, self.gred1_My, label='Upogibni moment')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('My [Nm]')
        plt.xlim(0, self.gred1_l1+self.gred1_l2)
        plt.grid()

    # --------Statika gredi 2--------
    def statika_gredi_2(self, l1, l2, l3):
        """Metoda sprejme naslednje podatke: razdalja od podpore A do središča 2. zobnika - l1 [mm], razdalja od središča 2. do središča 3. zobnika 
        - l2 [mm], razdalja od središča 3. zobinka do podpore B - l3 [mm].
            
        Izvedejo se preračuni naslednjih veličin: Tangencialna sila - Ft2 in Ft3 [N], radialna sila - Fr2 in Fr3 [N] in če je uporabljen zobnik s 
        poševnim ozobjem pe aksialna sila - Fa2 in Fa3 [N]. Sile na ležaje v podpori A: Ay [N], Az [N] ter v podpori B:  By [N], Bz [N]. Skupna obremenitev 
        ležaja v podpori A - A [N] in v podpori B - B[N]. Največji upogibni moment - Mmax [Nm].

        Ob klicanju metode se prav tako izrišejo grafi prečne sile in upogibnega momenta v X-Y in X-Z ravnini."""

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.Ft2 = 2 * self.T2 / (self.di()[2] / 1000)
        self.Ft3 = 2 * self.T2 / (self.di()[3] / 1000)
        self.Fr2 = self.Ft2 * np.tan(20 * np.pi / 180)
        self.Fr3 = self.Ft3 * np.tan(20 * np.pi / 180)
        if self.beta1 != 0:
            self.Fa2 = self.Ft2 * np.tan(self.beta1 * np.pi / 180)
        if self.beta2 != 0:
            self.Fa3 = self.Ft3 * np.tan(self.beta2 * np.pi / 180)

        # XY ravnina
        # Ay + Fr3 = By + Fr2
        # Fr2 * l1 - Fr3 * (l1 + l2)  + By * (l1 + l2 + l3) = 0
        A1 = np.array([[1, -1],
                      [0, (self.l1 + self.l2 + self.l3)]])

        b1 = np.array([self.Fr2 - self.Fr3, self.Fr3 * (self.l1 + self.l2) - self.Fr2 * self.l1])

        self.gred2_Ay = np.linalg.solve(A1, b1)[0]
        self.gred2_By = np.linalg.solve(A1, b1)[1]

        # XZ ravnina
        # Az + Bz = Ft3 + Ft2
        # By * (l1 + l2 + l3) = Ft3 * (l1 + l2) + Ft2 * l1
        A2 = np.array([[1, 1],
                      [0, (self.l1 + self.l2 + self.l3)]])
        b2 = np.array([self.Ft2 + self.Ft3, self.Ft3 * (self.l1 + self.l2) + self.Ft2 * self.l1])

        self.gred2_Az = np.linalg.solve(A2, b2)[0]
        self.gred2_Bz = np.linalg.solve(A2, b2)[1]

        self.statika_gredi_2_graf_xy_T()
        self.statika_gredi_2_graf_xy_M()
        self.statika_gredi_2_graf_xz_T()
        self.statika_gredi_2_graf_xz_M()
        plt.tight_layout()
        plt.show()

        self.gred2_A = (self.gred2_Ay ** 2 + self.gred2_Az ** 2) ** (1 / 2)
        self.gred2_B = (self.gred2_By ** 2 + self.gred2_Bz ** 2) ** (1 / 2)
        self.gred2_Mmax = (self.gred2_Mymax ** 2 + self.gred2_Mzmax ** 2) ** (1 / 2)

        self.gredi[2]['statika']['M_max'] = {1: (self.gredi[2]['statika']['Mz_max'][1]**2 +  self.gredi[2]['statika']['My_max'][1]**2)**(1/2),
                                             2: (self.gredi[2]['statika']['Mz_max'][2]**2 +  self.gredi[2]['statika']['My_max'][2]**2)**(1/2)}

        # Potek momentov
        self.gred2_M = (self.gred2_My ** 2 + self.gred2_Mz ** 2) ** (1 / 2)

    def statika_gredi_2_graf_xy_T(self):
        """Metoda za izris poteka prečne sile na gredi 2 v X-Y ravnini."""
        x = np.arange(0, self.l1+self.l2+self.l3, 0.1)
        T1 = [-self.gred2_Ay for _ in x[0: np.nonzero(x == self.l1)[0][0]]]
        T2 = [self.Fr2 - self.gred2_Ay for _ in x[np.nonzero(x == self.l1)[0][0]: np.nonzero(x == self.l1 + self.l2)[0][0]]]
        T3 = [-self.gred2_By for _ in x[np.nonzero(x == self.l1 + self.l2)[0][0]:]]
        T = np.concatenate((T1, T2, T3), 0)

        self.gredi[2] = {'statika': {'Ty': {1: max(T1), 2: max(T2), 3: max(T3)}}}

        plt.subplot(221)
        plt.title('Graf prečne sile Ty druge gredi')
        plt.plot(x, T, label='Prečna sila')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Ty [N]')
        plt.xlim(0, self.l1+self.l2+self.l3)
        plt.grid()

    def statika_gredi_2_graf_xy_M(self):
        """Metoda za izris poteka upogibnega momenta na gredi 2 v X-Y ravnini."""
        x = np.arange(0, self.l1+self.l2+self.l3, 0.1)
        M1 = [-self.gred2_Ay * _ for _ in x[0: np.nonzero(x == self.l1)[0][0]]]
        M2 = [self.Fr2 * (_-self.l1) - self.gred2_Ay * _ for _ in x[np.nonzero(x == self.l1)[0][0]: np.nonzero(x == self.l1 + self.l2)[0][0]]]
        M3 = [self.Fr2 * (_-self.l1) - self.gred2_Ay * _ - self.Fr3 * (_ - self.l1 - self.l2) for _ in x[np.nonzero(x == self.l1 + self.l2)[0][0]:]]
        self.gred2_Mz = np.concatenate((M1, M2, M3), 0) / 1000

        self.gred2_Mzmax = max(abs(self.gred2_Mz))
        self.gredi[2]['statika']['Mz_max'] = {1: max(map(abs, M1))/1000, 2:max(map(abs, M2))/1000}

        plt.subplot(223)
        plt.title('Graf momenta Mz druge gredi')
        plt.plot(x, self.gred2_Mz, label='Upogibni moment')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Mz [Nm]')
        plt.xlim(0, self.l1+self.l2+self.l3)
        plt.grid()

    def statika_gredi_2_graf_xz_T(self):
        """Metoda za izris poteka prečne sile na gredi 2 v X-Z ravnini."""
        x = np.arange(0, self.l1+self.l2+self.l3, 0.1)
        T1 = [self.gred2_Az for _ in x[0: np.nonzero(x == self.l1)[0][0]]]
        T2 = [self.gred2_Az - self.Ft2 for _ in x[np.nonzero(x == self.l1)[0][0]: np.nonzero(x == self.l1 + self.l2)[0][0]]]
        T3 = [-self.gred2_Bz for _ in x[np.nonzero(x == self.l1 + self.l2)[0][0]:]]
        T = np.concatenate((T1, T2, T3), 0)

        self.gredi[2]['statika']['Tz'] = {1: max(T1), 2: max(T2), 3: max(T3)}

        plt.subplot(222)
        plt.title('Graf prečne sile Tz druge gredi')
        plt.plot(x, T, label='Prečna sila')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Tz [N]')
        plt.xlim(0, self.l1+self.l2+self.l3)
        plt.grid()

    def statika_gredi_2_graf_xz_M(self):
        """Metoda za izris poteka upogibnega momenta na gredi 2 v X-Z ravnini."""
        x = np.arange(0, self.l1+self.l2+self.l3, 0.1)
        M1 = [self.gred2_Az * _ for _ in x[0: np.nonzero(x == self.l1)[0][0]]]
        M2 = [-self.Ft2 * (_-self.l1) + self.gred2_Az * _ for _ in x[np.nonzero(x == self.l1)[0][0]: np.nonzero(x == self.l1 + self.l2)[0][0]]]
        M3 = [-self.Ft2 * (_-self.l1) + self.gred2_Az * _ - self.Ft3 * (_ - self.l1 - self.l2) for _ in x[np.nonzero(x == self.l1 + self.l2)[0][0]:]]
        self.gred2_My = np.concatenate((M1, M2, M3), 0) / 1000

        self.gred2_Mymax = max(abs(self.gred2_My))
        self.gredi[2]['statika']['My_max'] = {1: max(map(abs, M1))/1000, 2: max(map(abs, M2))/1000}

        plt.subplot(224)
        plt.title('Graf momenta My druge gredi')
        plt.plot(x, self.gred2_My, label='Upogibni moment')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('My [Nm]')
        plt.xlim(0, self.l1+self.l2+self.l3)
        plt.grid()

    # --------Statika gredi 3--------
    def statika_gredi_3(self, l1, l2):
        """Metoda sprejme naslednje podatke: razdalja od podpore A do središča 4. zobnika predstavlja seštevek dolžin l1 in l2 [mm]  (njujino 
        razmerje nima vpliva, pomembna le skupna dolžina), razdalja od središča zobnika 4. do  podpore B - l3 [mm].

        Izvedejo se preračuni naslednjih veličin: Tangencialna sila - Ft4 [N], radialna sila - Fr4 [N] in če je uporabljen zobnik s poševnim 
        ozobjem pe aksialna sila - Fa4 [N]. Sile na ležaje v podpori A: Ay [N], Az [N] ter v podpori B:  By [N], Bz [N]. Skupna obremenitev ležaja 
        v podpori A - A [N] in v podpori B - B[N]. Največji upogibni moment - Mmax [Nm].

        Ob klicanju metode se prav tako izrišejo grafi prečne sile in upogibnega momenta v X-Y in X-Z ravnini."""

        self.gred3_l1 = l1
        self.gred3_l2 = l2
        self.Ft4 = 2 * self.T3 / (self.di()[1] / 1000)
        self.Fr4 = self.Ft4 * np.tan(20 * np.pi / 180)
        if self.beta2 != 0:
            self.Fa4 = self.Ft4 * np.tan(self.beta2 * np.pi / 180)  # beta - kot posevnosti

        # XY ravnina
        # Ay + By = Ft1
        # Fr4 * (l1 + l2) = By * (l1 + l2 + l3)
        A1 = np.array([[1, 1],
                       [0, (l1 + l2)]])
        b1 = np.array([self.Fr4, self.Fr4 * (self.gred3_l1)])

        self.gred3_Ay = np.linalg.solve(A1, b1)[0]
        print(self.gred3_Ay)
        print(self.Fr4)
        self.gred3_By = np.linalg.solve(A1, b1)[1]

        # XZ ravnina
        # Az + Bz = Ft1
        # Ft4 * (l1 + l2) = Bz * (l1 + l2 + l3)
        A2 = np.array([[1, 1],
                       [0, (l1 + l2)]])
        b2 = np.array([self.Ft4, self.Ft4 * (self.gred3_l1)])

        self.gred3_Az = np.linalg.solve(A2, b2)[0]
        self.gred3_Bz = np.linalg.solve(A2, b2)[1]

        self.statika_gredi_3_graf_xy_T()
        self.statika_gredi_3_graf_xy_M()
        self.statika_gredi_3_graf_xz_T()
        self.statika_gredi_3_graf_xz_M()
        plt.tight_layout()
        plt.show()

        self.gred3_A = (self.gred3_Ay ** 2 + self.gred3_Az ** 2) ** (1 / 2)
        self.gred3_B = (self.gred3_By ** 2 + self.gred3_Bz ** 2) ** (1 / 2)
        self.gred3_Mmax = (self.gred3_Mymax ** 2 + self.gred3_Mzmax ** 2) ** (1 / 2)

        self.gredi[3]['statika']['M_max'] = self.gred3_Mmax

        # Potek momentov
        self.gred3_M = (self.gred3_My ** 2 + self.gred3_Mz ** 2) ** (1 / 2)

    def statika_gredi_3_graf_xy_T(self):
        """Metoda za izris poteka prečne sile na gredi 3 v X-Y ravnini."""
        x = np.arange(0, self.gred3_l1 + self.gred3_l2, 0.1)
        T1 = [-self.gred3_Ay for _ in x[0: np.nonzero(x == self.gred3_l1)[0][0]]]
        T2 = [self.gred3_By for _ in x[np.nonzero(x == (self.gred3_l1))[0][0]:]]
        T = np.concatenate((T1, T2), 0)

        self.gredi[3] = {'statika': {'Ty': {1: max(T1), 2: max(T2)}}}

        plt.subplot(221)
        plt.title('Graf prečne sile Ty tretje gredi')
        plt.plot(x, T, label='Prečna sila')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Ty [N]')
        plt.xlim(0, self.gred3_l1 + self.gred3_l2)
        plt.grid()

    def statika_gredi_3_graf_xy_M(self):
        """Metoda za izris poteka upogibnega momenta na gredi 3 v X-Y ravnini."""
        x = np.arange(0, self.gred3_l1 + self.gred3_l2, 0.1)
        M1 = [-self.gred3_Ay * _ for _ in x[0: np.nonzero(x == (self.gred3_l1))[0][0]]]
        M2 = [self.Fr4 * (_ - (self.gred3_l1)) - self.gred3_Ay * _ for _ in x[np.nonzero(x == (self.gred3_l1))[0][0]:]]
        self.gred3_Mz = np.concatenate((M1, M2), 0) / 1000

        self.gred3_Mzmax = max(abs(self.gred3_Mz))
        self.gredi[3]['statika']['Mz_max'] = self.gred3_Mzmax

        plt.subplot(223)
        plt.title('Graf momenta Mz tretje gredi')
        plt.plot(x, self.gred3_Mz, label='Upogibni moment')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Mz [Nm]')
        plt.xlim(0, self.gred3_l1 + self.gred3_l2)
        plt.grid()

    def statika_gredi_3_graf_xz_T(self):
        """Metoda za izris poteka prečne sile na gredi 3 v X-Z ravnini."""
        x = np.arange(0, self.gred3_l1 + self.gred3_l2, 0.1)
        T1 = [-self.gred3_Az for _ in x[0: np.nonzero(x == (self.gred3_l1))[0][0]]]
        T2 = [self.gred3_Bz for _ in x[np.nonzero(x == (self.gred3_l1))[0][0]:]]
        T = np.concatenate((T1, T2), 0)

        self.gredi[3]['statika']['Tz'] = {1: max(T1), 2: max(T2)}

        plt.subplot(222)
        plt.title('Graf prečne sile Tz tretje gredi')
        plt.plot(x, T, label='Prečna sila')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('Tz [N]')
        plt.xlim(0, self.gred3_l1 + self.gred3_l2)
        plt.grid()

    def statika_gredi_3_graf_xz_M(self):
        """Metoda za izris poteka upogibnega momenta na gredi 3 v X-Z ravnini."""
        x = np.arange(0, self.gred3_l1 + self.gred3_l2, 0.1)
        M1 = [-self.gred3_Az * _ for _ in x[0: np.nonzero(x == (self.gred3_l1))[0][0]]]
        M2 = [self.Ft4 * (_ - (self.gred3_l1)) - self.gred3_Az * _ for _ in x[np.nonzero(x == (self.gred3_l1))[0][0]:]]
        self.gred3_My = np.concatenate((M1, M2), 0) / 1000

        self.gred3_Mymax = max(abs(self.gred3_My))
        self.gredi[3]['statika']['My_max'] = self.gred3_Mymax

        plt.subplot(224)
        plt.title('Graf momenta My tretje gredi')
        plt.plot(x, self.gred3_My, label='Upogibni moment')
        plt.axhline(color='r')
        plt.xlabel('x [mm]')
        plt.ylabel('My [Nm]')
        plt.xlim(0, self.gred3_l1 + self.gred3_l2)
        plt.grid()

    # --------Laboratorisjka vaja--------
    def LV_preberi_meritve(self, pot):
        """Metoda kot vhodni parameter sprejme le pot do tekstovne datoteke, v kateri so rezultati meritev laboratorijske vaje.
        Iz datoteke določi naslednje veličine: čas t[s], Mskl[Nm], Mzav[Nm], nem[s^1], nskl[s^1]. 
        Izračunajo se: omegaskl[rad/s], omegaem[rad/s], Mpos[Nm], Ptr[W], Ppos[W], LV_Wtr[J], LV_Wpos[J] in LV_Jc[mm^4]."""

        # Uvozimo podatke iz datoteke in iz njih preberemo vrednosti
        self.podatkiLV = np.loadtxt(pot, delimiter='\t', skiprows=1, dtype=float)
        self.t = self.podatkiLV[:, 0]
        self.Mskl = (self.podatkiLV[:, 1]) / 100
        self.Mzav = (self.podatkiLV[:, 2]) / 100
        self.nem = (self.podatkiLV[:, 3]) / 60 + 0.00001
        self.nskl = (self.podatkiLV[:, 4]) / 60 + 0.00001

        # Izračunamo potek kotne hitrosti elektromotorja in sklopke
        self.omegaem = 2 * np.pi * self.nem
        self.omegaskl = 2 * np.pi * self.nskl

        # Izračunamo moment pospeševanja, moč trenja in moč pospeševanja
        self.Mpos = self.Mskl - self.Mzav
        self.Ptr = self.Mskl * (self.omegaem - self.omegaskl)
        self.Ppos = self.Mpos * self.omegaskl

        # Z integracijo izračunamo delo trenja, delo pospeševanja ter vztrajnostni moment sklopke
        self.LV_Wtr = np.trapz(self.Ptr, self.t)
        self.LV_Wpos = np.trapz(self.Ppos, self.t)
        self.LV_Jc = 2 * self.LV_Wpos / self.omegaskl ** 2

        # Izračunamo maks. dovoljeni čas drsenja
        self.LV_tdrsenja = self.LV_Wtr / max(self.Ptr)

    def LV_graf1(self):
        """"Metoda izriše graf odvisnosti vrtilnih hitrosti od časa za elektromotor in sklopko."""

        plt.title('Graf odvisnosti vrtilnih hitrosti od časa')
        plt.plot(self.t, self.nem, label='Elektromotor')
        plt.plot(self.t, self.nskl, label='Sklopka')
        plt.xlabel('Čas [s]')
        plt.ylabel('Vrtilna hitrost [1/s]')
        plt.legend()
        plt.grid()

    def LV_graf2(self):
        """"Metoda izriše graf odvisnosti momentov sklopke in zavore od časa."""

        plt.title('Graf odvisnosti momentov sklopke in zavore od časa')
        plt.plot(self.t, self.Mskl, label='Sklopka - trenje')
        plt.plot(self.t, self.Mzav, label='Zavora - koristno')
        plt.xlabel('Čas [s]')
        plt.ylabel('Moment [Nm]')
        plt.legend()
        plt.grid()

    def LV_graf3(self):
        """"Metoda izriše graf odvisnosti moči trenja in pospeševanja od časa."""

        plt.title('Graf odvisnoti moči trenja in pospeševanja od časa')
        plt.plot(self.t, self.Ptr, label='Trenje')
        plt.plot(self.t, self.Ppos, label='Pospeševanje')
        plt.xlabel('Čas [s]')
        plt.ylabel('Moč [W]')
        plt.legend(loc=(0.25, 0.75))
        plt.xlim(0,17)
        plt.grid()

    def LV_TsklR(self, Tok=22.8, Tkon=28.2):
        """"Metoda vrne realno razliko temperatur na splopki (pred in po zagonu)."""

        self.Tok = Tok
        self.Tkon = Tkon
        return abs(Tkon - Tok)

    def LV_TsklT(self, mpl=8.3, cp=460):
        """"Metoda sprejme maso plošče [kg] in cp materiala [J/kgK] ter vrne teoretično razliko temperatur na sploki (pred in po zagonu)."""

        return self.LV_Wtr / (mpl * cp)

    def LV_nzagonov(self, Tdop=373):
        """Sprejme argument Tdop[°C], vrne seznam, pri čemer prva vrednost predstavlja št. zagonov ob upoštevanju realne razlike temperatur,
         drugi pa št. zagonov ob upoštevanju teoretične razlike temperatur."""

        return [(Tdop - self.Tok) / self.LV_TsklR(), (Tdop - self.Tok) / self.LV_TsklT()]

    # --------Lezaji--------
    lezaji = {} # Pripravimo slovar, v katerega bomo dodajali lastnosti ležajev za vse 3 gredi

    def lezaji_staticna(self, gred, C0, fiksen = 'A', vrsta=[1,1]):
        """Metoda sprejme naslednje podatke:
         - gred (za katero gred zelimo izracunati stacicni preracun lezajev)
         - fiksen (kateri ležaj, levi ali desni je fiksen)
         - C0 (slovar statičnih nosilnosts ležajev [N])
         - vrsta ([i, j] i=1-kroglični ležaji; 2-valjčni,stozčasti ali sodčkasti ležaji; 3-aksialni sferični ležaji
                         j=1-normalno obratovanje; 2-sunkovita obremenitev; 3-zahtevan zelo miren tek)"""

        self.C0 = C0

        # Priprava praznih slovajev
        self.X0 = {}
        self.Y0 = {}
        self.P0 = {}
        self.S0 = {}

        # Določitev ustreznih Fa in Fr za oba ležaja glede na izbrano gred in fiksen ležaj
        if fiksen == 'A': self.prost = 'B'
        if fiksen == 'B': self.prost = 'A'

        if gred == 1:
            #self.X0[1]={}
            Fr = {'A' : self.gred1_A, 'B' : self.gred1_B}
            if self.beta1 != 0:
                Fa = {fiksen : self.Fa1, self.prost : 0}
            else:
                Fa = {fiksen : 0, self.prost : 0}

        if gred == 2:
            Fr = {'A' : self.gred2_A, 'B' : self.gred2_B}
            if self.beta1 != 0:
                Fa = {fiksen : self.Fa2, self.prost : 0}
            else:
                Fa = {fiksen : 0, self.prost : 0}

        if gred == 3:
            Fr = {'A' : self.gred3_A, 'B' : self.gred3_B}
            if self.beta2 != 0:
                Fa = {fiksen : self.Fa4, self.prost : 0}
            else:
                Fa = {fiksen : 0, self.prost : 0}

        # Tabela 6, str. 22
        for lez in ['A', 'B']:
            if Fa[lez] / Fr[lez] <= 0.8:
                self.X0[lez] = 1
                self.Y0[lez] = 0
            if Fa[lez] / Fr[lez] > 0.8:
                self.X0[lez] = 0.6
                self.Y0[lez] = 0.5

        # Izračun P0 in S0 za oba ležaja
        for lez in ['A', 'B']:
            self.P0[lez] = self.X0[lez] * Fr[lez] + self.Y0[lez] * Fa[lez]
            self.S0[lez] = self.C0[lez] / self.P0[lez]

        # Tabela 7, str. 22
        if vrsta[0] == 1:
            if vrsta[1] == 1: self.S0min = 1
            if vrsta[1] == 2: self.S0min = 1.5
            if vrsta[1] == 3: self.S0min = 2
        if vrsta[0] == 2:
            if vrsta[1] == 1: self.S0min = 1.5
            if vrsta[1] == 2: self.S0min = 2
            if vrsta[1] == 3: self.S0min = 3
        if vrsta[0] == 3: self.S0min = 4

        # Izpis, če ležaja zdržita dano obremenitev
        for lez in ['A', 'B']:
            if self.S0[lez] >= self.S0min:
                print(f'Statična varnost S0 je večja od dovoljene, zato ležaj v podpori {lez:s} zdrži statično obremenitev.')
            if self.S0[lez] < self.S0min:
                print(f'Statična varnost S0 je manjša od dovoljene, zato ležaj v podpori {lez:s} ne zdrži statične obremenitve.')

        # Izracune za trenutno gred dodamo v skupen slovar
        self.lezaji[gred] = {'P0': {'A': self.P0['A'], 'B': self.P0['B']},
                             'S0': {'A': self.S0['A'], 'B': self.S0['B']},
                             'S0min': {'A': self.S0min, 'B': self.S0min}}

    def lezaji_doba_trajana(self, gred, C, fiksen = 'A'):
        """Metoda sprejme naslednje podatke:
             - gred (za katero gred zelimo izracunati stacicni preracun lezajev)
             - fiksen (kateri ležaj, levi ali desni je fiksen)
             - C (dinamična nosilnost ležaja [N]"""

        # Določitev ustreznih Fa in Fr za oba ležaja glede na izbrano gred in fiksen ležaj
        if fiksen == 'A': self.prost = 'B'
        if fiksen == 'B': self.prost = 'A'

        if gred == 1:
            n_ = self.n1
            Fr = {'A' : self.gred1_A, 'B' : self.gred1_B}
            if self.beta1 != 0:
                Fa = {fiksen : self.Fa1, self.prost : 0}
            else:
                Fa = {fiksen : 0, self.prost : 0}

        if gred == 2:
            n_ = (self.omega2 / (2 * np.pi)) * 60
            Fr = {'A' : self.gred2_A, 'B' : self.gred2_B}
            if self.beta1 != 0:
                Fa = {fiksen : self.Fa2, self.prost : 0}
            else:
                Fa = {fiksen : 0, self.prost : 0}

        if gred == 3:
            n_ = (self.omega3 / (2 * np.pi)) * 60
            Fr = {'A' : self.gred3_A, 'B' : self.gred3_B}
            if self.beta2 != 0:
                Fa = {fiksen : self.Fa4, self.prost : 0}
            else:
                Fa = {fiksen : 0, self.prost : 0}

        # Priprava praznih slovajev
        self.X = {}
        self.Y = {}
        self.P = {}
        self.L10 = {}
        self.L10h = {}

        # Podatki iz tabele 8, str. 23
        e = [0.22, 0.23, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.39, 0.43]
        FapoC0 = [0.025, 0.0325, 0.04, 0.055, 0.07, 0.1, 0.13, 0.19, 0.25, 0.375, 0.5]
        seznamY = [2, 1.9, 1.8, 1.69, 1.58, 1.49, 1.4, 1.3, 1.2, 1.1, 1]

        # Tabela 8, str. 23
        for lez in ['A', 'B']:
            for _, n, y in zip(e, FapoC0, seznamY):
                if n - 0.01 < Fa[lez] / self.C0[lez] < n + 0.01 and Fa[lez] / Fr[lez] <= _:
                    self.X[lez] = 1
                    self.Y[lez] = 0
                    break
                if n - 0.01 < Fa[lez] / self.C0[lez] < n + 0.01 and Fa[lez] / Fr[lez] > _:
                    self.X[lez] = 0.56
                    self.Y[lez] = y
                    break
                else:
                    self.X[lez] = 1
                    self.Y[lez] = 0

        # Izračun P, L10 in L10h za oba ležaja
        for lez in ['A', 'B']:
            self.P[lez] = self.X[lez] * Fr[lez] + self.Y[lez] * Fa[lez]
            self.L10[lez] = (C[lez] / self.P[lez]) ** 3 * 10 ** 6
            self.L10h[lez] = self.L10[lez] / (60 * n_)

        # Izracune za trenutno gred dodamo v skupen slovar
        self.lezaji[gred]['P'] = {'A': self.P['A'], 'B': self.P['B']}
        self.lezaji[gred]['L10'] = {'A': self.L10['A'], 'B': self.L10['B']}
        self.lezaji[gred]['L10h'] = {'A': self.L10h['A'], 'B': self.L10h['B']}

    def lezaji_razsirjena_doba_trajana(self, gred, T_obr, Cu, d_lez, D_lez, olje = 1500, verjetnost_neposkodbe = 90):
        """Metoda sprejme naslednje podatke:
                     - T_obr [°C] (predvidena obratovalna temperatura olja)
                     - olje (ISO VG ___), privzeta vrednost je 1500
                     - verjentost_neposkodbe [%] (verjetnost, da nebo prišlo do poškodbe)"""

        self.n_hitrosti = {1:self.n1, 2:(self.omega2 / (2 * np.pi)) * 60, 3:(self.omega3 / (2 * np.pi)) * 60}

        # Tabela 9
        tabela_9 = {90:1, 95:0.64, 96:0.55, 97:0.47, 98:0.37, 99:0.25, 99.4:0.19, 99.8:0.12, 99.95:0.077}
        self.a1 = tabela_9[verjetnost_neposkodbe]

        # Zaženemo metodo, ki nam vrne interpolacijo diagrama 10
        self.diagram_10()

        self.lezaji_din_visk = self.dia10_funkc[olje](T_obr) # dinamicna viskosnot olja pri izbranenm olju in T obratovanja
        self.lezaji_kin_visk = self.lezaji_din_visk * 1.1111  # kinematicna viskosnot olja pri izbranenm olju in T obratovanja
        #print(f'gred {gred}: visk = {self.lezaji_kin_visk}')

        #---------------------------------------------------------------------
        # Diagram 11
        # Prebrani podatki
        zac_vr1 = {100000: 4, 50000: 6.5, 20000: 10, 10000: 7.5, 5000: 20.5, 3000: 30, 2000: 36, 1500: 38, 1000: 45,
                   900: 47, 800: 52, 600: 70, 500: 80, 400: 100, 300: 140, 200: 180, 100: 310, 50: 540}
        zac_vr = {}
        for i in zac_vr1.keys():
            zac_vr[i] = np.log(zac_vr1[i]) + 0.5 * np.log(10)

        # Pripravimo funkcije
        self.dia11_funkc = {}
        for n_ in zac_vr.keys():
            self.dia11_funkc[n_] = lambda x, n=zac_vr[n_]: np.exp(-0.5 * np.log(x) + n)

        # Razlike dolocenih hitrosti in dejanske hitrosti ter najblizja dolocena hitrost
        razlike_n = []
        for n_prim in list(zac_vr1.keys()):
            razlike_n.append(abs(n_prim - self.n_hitrosti[gred]))
        n_cca = list(zac_vr1.keys())[razlike_n.index(min(razlike_n))]

        # Dolocimo potrebno kinematicno viskoznost v1
        self.lezaji_kin_visk_potrebna = {}
        for lez in ['A', 'B']:
            self.lezaji_kin_visk_potrebna[lez] = self.dia11_funkc[n_cca]((D_lez[lez] + d_lez[lez])/2)
        # ---------------------------------------------------------------------

        # Diagram 12
        # Zaženemo metodo, ki nam vrne interpolacijo diagrama 10
        self.diagram_12()

        # Dejanski kappa
        kappa = {}
        for lez in ['A', 'B']:
            kappa[lez] = self.lezaji_kin_visk/self.lezaji_kin_visk_potrebna[lez]
            #print(f'gred {gred}: kappa = {kappa[lez]}')

        # Najblizja vrednost kappe iz diagrama (glede na dejansko)
        kappa_cca = {}
        for lez in ['A', 'B']:
            razlike_kappa = []
            for kappa_prim in list(self.funkc_12.keys()):
                razlike_kappa.append(abs(kappa_prim - kappa[lez]))
            kappa_cca[lez] = list(self.funkc_12.keys())[razlike_kappa.index(min(razlike_kappa))]

        ecCu = {}
        for lez in ['A', 'B']:
            ecCu[lez] = 0.5 * Cu[lez] / self.lezaji[gred]['P'][lez]

        # Dolocimo aiso za vsak lezaj posebej
        self.lezaji[gred]['aiso'] = {'A': self.funkc_12[kappa_cca['A']](ecCu['A']), 'B': self.funkc_12[kappa_cca['B']](ecCu['B'])}

        #  Dolocimo Lna za vsak lezaj posebej
        Lna = {}
        Lna_h = {}
        for lez in ['A', 'B']:
            Lna[lez] = self.a1 * self.lezaji[gred]['aiso'][lez] * self.lezaji[gred]['L10'][lez]
            Lna_h[lez] = self.a1 * self.lezaji[gred]['aiso'][lez] * self.lezaji[gred]['L10h'][lez]

        # In ga priredimo glavnemu slovarju za sezname
        self.lezaji[gred]['Lna'] = {'A': Lna['A'], 'B': Lna['B']}
        self.lezaji[gred]['Lna_h'] = {'A': Lna_h['A'], 'B': Lna_h['B']}
        self.lezaji[gred]['kappa'] = {'A': kappa['A'], 'B': kappa['B']}
        self.lezaji[gred]['kappa_cca'] = {'A': kappa_cca['A'], 'B': kappa_cca['B']}

    def diagram_10(self):
        """Metoda vrne diagram 10."""

        # Prebrane točke iz grafa
        diagram_10_navaden = {1500: [[20,10000],[60,320],[80,100],[100,40],[160,7.1]],
                              1000: [[20,6200],[40,1000],[100,30],[140,9.8],[160,6.4]],
                              680: [[20,4100],[40,700],[95,30],[140,8],[160,5.6]],
                              460: [[20,2500],[40,490],[80,45],[100,20],[160,4.9]],
                              320: [[20,1600],[40,300],[80,33],[120,9],[160,4]],
                              220: [[20,1000],[60,70],[110,10],[155,4],[160,3.7]],
                              150: [[20,630],[58,50],[80,20],[120,6.2],[160,3.1]],
                              100: [[20,370],[40,100],[100,8],[140,3.6],[160,2.8]],
                              68: [[20,230],[60,26],[100,6.7],[140,3],[160,2.45]],
                              46: [[20,145],[60,20],[80,9],[120,3.6],[160,2.1]],
                              32: [[20,85],[60,15],[100,4.1],[120,3],[160,1.9]],
                              22: [[20,52],[60,10],[100,3.5],[140,2],[160,1.75]],
                              15: [[20,32],[60,7],[100,2.9],[140,1.8],[160,1.6]],
                              10: [[20,19],[60,5],[100,2.2],[140,1.5],[160,1.45]],
                              7: [[20,14],[60,3.5],[100,1.8],[140,1.3],[160,1.1]],
                              5: [[20,9],[60,2.8],[100,1.6],[140,1.1],[160,1]],
                              3: [[20,3.8],[60,2.1],[100,1.4],[140,1.01],[160,1]],
                              2: [[20,2.2],[60,1.6],[100,1.2],[140,1],[160,1]]}

        # Enak slovar, le z numpy seznami
        diagram_10 = {}
        for olje in diagram_10_navaden.keys():
            diagram_10[olje] = np.array(diagram_10_navaden[olje])

        # Pripravimo prazen slovar za interpolacijske funkcije
        self.dia10_funkc = {}

        # Za vsako olje interpoliramo čez podane točke in dobljeno funkcijo shranimo v pripravljen slovar
        for oil in diagram_10.keys():
            funkc_zac = interp1d(diagram_10[oil][:, 0], np.log(diagram_10[oil][:, 1]), kind = 3)
            self.dia10_funkc[oil] = lambda x, funkc=funkc_zac: np.exp(funkc(x))  # znotraj zanke je lambdi potrebno določiti privzeto vrednost(funkcijo),
                                                                                 # in to tisto, ki jo želimo "lambifirati"

    def diagram_10_izris(self):
        """Metoda izriše diagram 10."""

        x_int = np.linspace(20, 160, 1000)

        for olje in self.dia10_funkc.keys():
            plt.plot(x_int, self.dia10_funkc[olje](x_int), label=olje)
        plt.yscale('log')
        plt.grid()
        plt.xlim(20, 160)
        plt.ylim(1, 10000)
        plt.legend(loc=(1, 0))
        plt.title('Diagram odvisnosti viskoznosti olja od Temp. obratovanja')
        plt.ylabel('Dinamična viskoznost olja')
        plt.xlabel('Temperatura obratovanja')
        plt.show()

    def diagram_12(self):
        """Metoda vrne diagram 12."""

        # Prebrane točke iz grafa
        diagram_12_raw = {4: [[0.005, 0.39], [0.01, 0.59], [0.1, 8], [0.2, 35]],
                          3: [[0.005, 0.38], [0.01, 0.53], [0.1, 5], [0.25, 50]],
                          2: [[0.005, 0.35], [0.01, 0.48], [0.1, 4.2], [0.31, 50]],
                          1.5: [[0.005, 0.31], [0.01, 0.43], [0.1, 3.5], [0.37, 50]],
                          1: [[0.005, 0.29], [0.01, 0.39], [0.1, 2.5], [0.48, 50]],
                          0.8: [[0.005, 0.25], [0.01, 0.31], [0.1, 1.6], [0.7, 50]],
                          0.6: [[0.005, 0.2], [0.01, 0.25], [0.1, 0.8], [1.5, 50]],
                          0.5: [[0.005, 0.18], [0.01, 0.21], [0.1, 0.42], [2, 25]],
                          0.4: [[0.005, 0.16], [0.01, 0.175], [0.1, 0.31], [5, 17]],
                          0.3: [[0.005, 0.15], [0.01, 0.16], [0.1, 0.26], [5, 5.5]],
                          0.2: [[0.005, 0.13], [0.01, 0.14], [0.1, 0.19], [5, 1.3]],
                          0.15: [[0.005, 0.12], [0.01, 0.125], [0.1, 0.15], [5, 0.5]]}

        # Enak slovar, le z numpy seznami
        self.diagram_12_podatki = {}
        for kapa in diagram_12_raw.keys():
            self.diagram_12_podatki[kapa] = np.array(diagram_12_raw[kapa])

        # Pripravimo prazen slovar za interpolacijske funkcije
        self.funkc_12 = {}

        # Za vsako kapo interpoliramo čez podane točke in dobljeno funkcijo shranimo v pripravljen slovar
        for kapa in self.diagram_12_podatki.keys():
            self.funkc_12[kapa] = interp1d(self.diagram_12_podatki[kapa][:, 0], self.diagram_12_podatki[kapa][:, 1], 2)

    # --------Vrednotenje gredi--------

    def kriticni_prerezi(self, gred, Rp02, Rm, Rmax, material, graf = None, tabela = None, sD = 2, ro=1):
        """Metoda izračuna, če določena gred zdrži obremenitve."""

        # Diagram 4, str. 9
        mi_k = 1 / (1 + (8 / ro) * (1 - Rp02 / Rm) ** 3)

        # Diagram 5, str. 9
        sigma_Df = {'S235': 180, 'S275': 210, 'E295': 250, 'E335': 300, 'E360': 350}
        tau_Dt = {'S235': 100, 'S275': 130, 'E295': 150, 'E335': 180, 'E360': 205}

        # Prireditev pravilnih vrednosti Mt glede na izbrano gred
        if gred == 1: Mt = self.T1;
        if gred == 2: Mt = self.T2;
        if gred == 3: Mt = self.T3;

        if tabela:
            if gred == 1: Mf = [self.gredi[1]['statika']['M_max'], self.gredi[1]['statika']['M_max']]
            if gred == 2: Mf = [self.gredi[2]['statika']['M_max'][1], self.gredi[2]['statika']['M_max'][2]]
            if gred == 3: Mf = [self.gredi[3]['statika']['M_max'], self.gredi[3]['statika']['M_max']]

            # Tabela 1, str. 8
            d = tabela['d_pod_zobnikom']
            if tabela['stevilka'] == 4: alfa_kf = 1.7; alfa_kt = 1.6; #krčni nased!
            if tabela['stevilka'] == 5: alfa_kf = 4; alfa_kt = 2.8;
            if tabela['stevilka'] == 10: alfa_kf = 4.2; alfa_kt = 3.6;
            if tabela['stevilka'] == 11: alfa_kf = 3.5; alfa_kt = 2.3;
            if tabela['stevilka'] == 12: alfa_kf = 2.9; alfa_kt = 2;

            beta_kf = 1 + mi_k * (alfa_kf - 1)
            beta_kt = 1 + mi_k * (alfa_kt - 1)

            # Maksimalna upogibna in torzijska napetost
            sigma_fmax = [beta_kf * 32 * Mf[0] / (np.pi * (d / 1000) ** 3) / (10 ** 6), beta_kf * 32 * Mf[1] / (np.pi * (d / 1000) ** 3) / (10 ** 6)]
            print(f"{gred} - {sigma_fmax}")
            tau_tmax = beta_kt * 16 * Mt / (np.pi * (d / 1000) ** 3) / (10 ** 6)
            alfa0 = sigma_Df[material] / (1.73 * tau_Dt[material])

            # Primerjalna napetost
            sigma_p = [(sigma_fmax[0] ** 2 + 3 * (alfa0 * tau_tmax) ** 2) ** (1 / 2), (sigma_fmax[1] ** 2 + 3 * (alfa0 * tau_tmax) ** 2) ** (1 / 2)]

        if graf:
            if gred == 1: Mf = self.gred1_M[graf["x_od_roba"]*10];
            if gred == 2: Mf = self.gred2_M[graf["x_od_roba"]*10];
            if gred == 3: Mf = self.gred3_M[graf["x_od_roba"]*10];

            # Diagram 1 in 2, str.7
            self.diagram_1()
            self.diagram_2()
            d = graf['d']
            dpoD = graf['d']/graf['D']
            t = (graf['D']-graf['d'])/2

            # Najblizja vrednost d/D iz diagrama (glede na dejansko)
            ropot = ro/t
            razlike_ropot1 = []
            for ropot_prim in list(self.funkc_1.keys()):
                razlike_ropot1.append(abs(ropot_prim - ropot))
            ropot_cca1 = list(self.funkc_1.keys())[razlike_ropot1.index(min(razlike_ropot1))]

            razlike_ropot2 = []
            for ropot_prim in list(self.funkc_2.keys()):
                razlike_ropot2.append(abs(ropot_prim - ropot))
            ropot_cca2 = list(self.funkc_2.keys())[razlike_ropot2.index(min(razlike_ropot2))]

            alfa_kf = self.funkc_1[ropot_cca1](dpoD)
            alfa_kt = self.funkc_2[ropot_cca2](dpoD)

            beta_kf = 1 + mi_k * (alfa_kf - 1)
            beta_kt = 1 + mi_k * (alfa_kt - 1)

            # Maksimalna upogibna in torzijska napetost
            sigma_fmax = beta_kf * 32 * Mf / (np.pi * (d / 1000) ** 3) / 10 ** 6
            tau_tmax = beta_kt * 16 * Mt / (np.pi * (d / 1000) ** 3) / 10 ** 6
            alfa0 = sigma_Df[material] / (1.73 * tau_Dt[material])

            # Primerjalna napetost
            sigma_p = (sigma_fmax** 2 + 3 * (alfa0 * tau_tmax) ** 2) ** (1 / 2)


        # Diagram 7, str.10
        podatki_7x = np.array([10, 40, 50, 70, 90, 120])
        podatki_7y = np.array([1, 0.85, 0.82, 0.77, 0.74, 0.72])
        funkc_7 = interp1d(podatki_7x, podatki_7y, 2)

        # Koeficient velikosti prereza b1
        b1 = funkc_7(d)

        # Zažene metodo, ki nam vrne diagram 8
        self.diagram_8()
        # Koeficient hrapavosti površine b2
        b2 = self.funkc_8[Rmax](Rm)

        # Dopustna napetost
        sigma_dop = sigma_Df[material] * b1 * b2 / sD

        # Ce seznam funkcijo za podano gred kličemo prvic, ustvari podslovar 'prerezi'
        if 'prerezi' not in self.gredi[gred].keys():
            self.gredi[gred]['prerezi'] = {'pod zobnikom': {}, 'stopnica': {}}

        if tabela:
            if gred != 2:
                if gred == 1:
                    self.gredi[gred]['prerezi']['pod zobnikom'] = {'sigma_fmax' : sigma_fmax[0], 'tau_tmax' : tau_tmax, 'sigma_p':sigma_p[0],
                                                                'sigma_dop': sigma_dop}
                if gred == 3:
                    self.gredi[gred]['prerezi']['pod zobnikom'] = {'sigma_fmax': sigma_fmax[0], 'tau_tmax': tau_tmax, 'sigma_p': sigma_p[0],
                                                                'sigma_dop': sigma_dop}
            else:
                self.gredi[gred]['prerezi']['pod zobnikom']['L'] = {'sigma_fmax': sigma_fmax[0], 'tau_tmax': tau_tmax, 'sigma_p': sigma_p[0],
                                                                    'sigma_dop': sigma_dop}
                self.gredi[gred]['prerezi']['pod zobnikom']['D'] = {'sigma_fmax': sigma_fmax[1], 'tau_tmax': tau_tmax, 'sigma_p': sigma_p[1],
                                                                    'sigma_dop': sigma_dop}

        if graf:
            if gred != 2:
                if gred == 1:
                    self.gredi[gred]['prerezi']['stopnica'] = {'sigma_fmax' : sigma_fmax, 'tau_tmax' : tau_tmax, 'sigma_p':sigma_p,
                                                                'sigma_dop': sigma_dop}
                if gred == 3:
                    self.gredi[gred]['prerezi']['stopnica'] = {'sigma_fmax': sigma_fmax, 'tau_tmax': tau_tmax, 'sigma_p': sigma_p,
                                                                'sigma_dop': sigma_dop}
            else:
                self.gredi[gred]['prerezi']['stopnica'] = {'sigma_fmax': sigma_fmax, 'tau_tmax': tau_tmax, 'sigma_p': sigma_p,
                                                            'sigma_dop': sigma_dop}
                self.gredi[gred]['prerezi']['stopnica']= {'sigma_fmax': sigma_fmax, 'tau_tmax': tau_tmax, 'sigma_p': sigma_p,
                                                            'sigma_dop': sigma_dop}

    def diagram_8(self):
        """Metoda vrne diagram 8."""

        self.podatki_8 = {2: np.array([[300, 0.985], [750, 0.95], [1400, 0.91]]),
                     4: np.array([[300, 0.97], [800, 0.895], [1400, 0.85]]),
                     6: np.array([[300, 0.95], [900, 0.85], [1400, 0.81]]),
                     10: np.array([[300, 0.92], [900, 0.8], [1400, 0.75]]),
                     20: np.array([[300, 0.9], [900, 0.75], [1400, 0.69]]),
                     40: np.array([[300, 0.85], [800, 0.705], [1400, 0.64]]),
                     100: np.array([[300, 0.82], [800, 0.66], [1400, 0.57]]),
                     'Lito': np.array([[300, 0.8], [600, 0.6], [810, 0.5]])}

        self.funkc_8 = {}

        for R in self.podatki_8.keys():
            self.funkc_8[R] = interp1d(self.podatki_8[R][:, 0], self.podatki_8[R][:, 1], 2)

    def diagram_8_izris(self):
        """Metoda izriše diagram 8."""

        for R in self.funkc_8.keys():
            x_int = np.linspace(300, max(self.podatki_8[R][:, 0]), 100)
            plt.plot(x_int, self.funkc_8[R](x_int), label=R)
        plt.xlim(300, 1400)
        plt.ylim(0.5, 1)
        plt.legend(loc=(1, 0))
        plt.xlabel('Natezna trdnost Rm [Mpa]')
        plt.ylabel('Koef. hrapavosti površine b2')
        plt.title('Diagram odvisnosti b2 od hrapavosti površine in natezne trdnosti')
        plt.grid()
        plt.show()

    def diagram_1(self):
        """Metoda vrne diagram 1."""

        self.podatki_1 = {0.07: np.array([[0.4, 2.75], [0.7, 3.4], [1, 5.2]]),
                         0.1: np.array([[0.4, 2.3], [0.7, 3], [1, 4.6]]),
                         0.15: np.array([[0.4, 2.1], [0.7, 2.6], [1, 3.8]]),
                         0.2: np.array([[0.4, 1.7], [0.7, 2.35], [1, 3.5]]),
                         0.3: np.array([[0.4, 1.6], [0.7, 2.15], [1, 3.1]]),
                         0.5: np.array([[0.4, 1.3], [0.7, 1.8], [1, 2.5]]),
                         1: np.array([[0.4, 1.2], [0.7, 1.5], [1, 2.1]]),
                         2: np.array([[0.4, 1.15], [0.7, 1.3], [1, 1.65]])}

        self.funkc_1 = {}

        for R in self.podatki_1.keys():
            self.funkc_1[R] = interp1d(self.podatki_1[R][:, 0], self.podatki_1[R][:, 1], 2)

    def diagram_1_izris(self):
        """Metoda izriše diagram 1."""

        for R in self.funkc_1.keys():
            x_int = np.linspace(0.4, 1, 100)
            plt.plot(x_int, self.funkc_1[R](x_int), label=R)
        plt.xlim(0.4, 1)
        plt.ylim(1, 6)
        plt.legend(loc=(1, 0))
        plt.ylabel('alfa_kf')
        plt.xlabel('d/D')
        plt.grid()
        plt.show()

    def diagram_2(self):
        """Metoda vrne diagram 2."""

        self.podatki_2 = {0.03:np.array([[0.4,2.2], [0.7,2.9],[1, 4.2]]),
                         0.04:np.array([[0.4,2],[0.7, 2.7],[1, 3.7]]),
                         0.06:np.array([[0.4,1.8],[0.7, 2.4],[1, 3.2]]),
                         0.1:np.array([[0.4,1.55],[0.7, 2],[1, 2.75]]),
                         0.2:np.array([[0.4,1.45],[0.7, 1.7],[1, 2.2]]),
                         0.4:np.array([[0.4,1.3],[0.7, 1.4],[1, 1.8]]),
                         1:np.array([[0.4,1.15],[0.7,1.25],[1, 1.5]])}

        self.funkc_2 = {}

        for R in self.podatki_2.keys():
            self.funkc_2[R] = interp1d(self.podatki_2[R][:, 0], self.podatki_2[R][:, 1], 2)

    def diagram_2_izris(self):
        """Metoda izriše diagram 2."""

        for R in self.funkc_2.keys():
            x_int = np.linspace(0.4, 1, 100)
            plt.plot(x_int, self.funkc_2[R](x_int), label=R)
        plt.xlim(0.4, 1)
        plt.ylim(1, 5)
        plt.legend(loc=(1, 0))
        plt.ylabel('alfa_kt')
        plt.xlabel('d/D')
        plt.grid()
        plt.show()

    # --------Povesi gredi--------

    def povesi_zasuki(self, gred, d, E, previs = None):
        """Metoda izračuna povese in zasuke za dano gred."""

        I =  np.pi * d**4 / 64

        if gred == 2:
            L = self.l1+self.l2+self.l3

            # X-Y ravnina
            # Prva sila
            F1 = self.Fr2
            x1 = np.arange(0, self.l1, 0.1)
            f1_xy = (F1 * L**3 /(6*E*I)) * ((x1/L)**2+(x1/L -2)*(x1/L))
            x2 = np.arange(self.l1, L, 0.1)
            f2_xy = (F1 * L ** 3 / (6 * E * I)) * ((x2 / L) ** 2 + (x2 / L - 2) * (x2 / L))
            alfa_A1_xy = np.arctan(F1 * self.l1 * (self.l2 + self.l3) * (L + (self.l2 + self.l3)) / (6 * E * I * L))
            alfa_B1_xy = np.arctan(F1 * self.l1 * (self.l2 + self.l3) * (L + self.l1) / (6 * E * I * L))

            # Druga sila
            F2 = self.Fr3
            x3 = np.arange(0, self.l2, 0.1)
            f3_xy = (F2 * L ** 3 / (6 * E * I)) * ((x3 / L) ** 2 + (x3 / L - 2) * (x3 / L))
            x4 = np.arange(self.l2, L, 0.1)
            f4_xy = (F2 * L ** 3 / (6 * E * I)) * ((x4 / L) ** 2 + (x4 / L - 2) * (x4 / L))
            alfa_A2_xy = np.arctan(F1 * (self.l1 + self.l2) * self.l3 * (L + self.l3) / (6 * E * I * L))
            alfa_B2_xy = np.arctan(F1 * (self.l1 + self.l2) * self.l3 * (L + (self.l1 + self.l2)) / (6 * E * I * L))

            # Skupni potek povesov in skupni zasuki
            f_xy = np.concatenate((f1_xy, f2_xy), 0) + np.concatenate((f3_xy, f4_xy), 0)
            alfa_A_xy = alfa_A1_xy + alfa_A2_xy
            alfa_B_xy = alfa_B1_xy + alfa_B2_xy

            # X-Z ravnina
            # Prva sila
            F1 = self.Ft2
            x1 = np.arange(0, self.l1, 0.1)
            f1_xz = (F1 * L ** 3 / (6 * E * I)) * ((x1 / L) ** 2 + (x1 / L - 2) * (x1 / L))
            x2 = np.arange(self.l1, L, 0.1)
            f2_xz = (F1 * L ** 3 / (6 * E * I)) * ((x2 / L) ** 2 + (x2 / L - 2) * (x2 / L))
            alfa_A1_xz = np.arctan(F1 * self.l1 * (self.l2 + self.l3) * (L + (self.l2 + self.l3)) / (6 * E * I * L))
            alfa_B1_xz = np.arctan(F1 * self.l1 * (self.l2 + self.l3) * (L + self.l1) / (6 * E * I * L))

            # Druga sila
            F2 = self.Ft3
            x3 = np.arange(0, self.l2, 0.1)
            f3_xz = (F2 * L ** 3 / (6 * E * I)) * ((x3 / L) ** 2 + (x3 / L - 2) * (x3 / L))
            x4 = np.arange(self.l2, L, 0.1)
            f4_xz = (F2 * L ** 3 / (6 * E * I)) * ((x4 / L) ** 2 + (x4 / L - 2) * (x4 / L))
            alfa_A2_xz = np.arctan(F1 * (self.l1 + self.l2) * self.l3 * (L + self.l3) / (6 * E * I * L))
            alfa_B2_xz = np.arctan(F1 * (self.l1 + self.l2) * self.l3 * (L + (self.l1 + self.l2)) / (6 * E * I * L))

            # Skupni potek povesov in skupni zasuki
            f_xz = np.concatenate((f1_xz, f2_xz), 0) + np.concatenate((f3_xz, f4_xz), 0)
            alfa_A_xz = alfa_A1_xz + alfa_A2_xz
            alfa_B_xz = alfa_B1_xz + alfa_B2_xz

            # Upostevanje obeh ravnin
            f = (f_xy ** 2 + f_xz ** 2) ** (1 / 2)
            alfa_A = (alfa_A_xy ** 2 + alfa_A_xz ** 2) ** (1 / 2)
            alfa_B = (alfa_B_xy ** 2 + alfa_B_xz ** 2) ** (1 / 2)

            # Poves pod zobnikoma in maksimalni
            f_pod_zobnikom1 = f[self.l1 * 10]
            f_pod_zobnikom2 = f[(self.l1 + self.l2)* 10]
            f_maksimalni = max(abs(f))

            self.gredi[gred]['povesi'] = {'pod zobnikom': {1: f_pod_zobnikom1, 2: f_pod_zobnikom2}, 'maksimalni': f_maksimalni}
            self.gredi[gred]['zasuki'] = {'A': alfa_A, 'B': alfa_B}

        if gred == 1:
            L = self.gred1_l1 + self.gred1_l2

            # X-Y ravnina
            F = self.Fr1
            # Potek povesov
            x1 = np.arange(0, self.gred1_l1, 0.1)
            f1 = (F * L ** 3 / (6 * E * I)) * ((x1 / L) ** 2 + (x1 / L - 2) * (x1 / L))
            x2 = np.arange(self.gred1_l1, L, 0.1)
            f2 = (F * L ** 3 / (6 * E * I)) * ((x2 / L) ** 2 + (x2 / L - 2) * (x2 / L))
            # Zasuki
            alfa_A = np.arctan(F * self.gred1_l1 * self.gred1_l2 * (L + self.gred1_l2) / (6 * E * I * L))
            alfa_B = np.arctan(F * self.gred1_l1 * self.gred1_l2 * (L + self.gred1_l1) / (6 * E * I * L))

            # Ce upostevamo se previs
            if previs:
                a = previs['a']
                F_p = previs['F']
                x_p = np.arange(0, L, 0.1)
                f_previs = -(F_p*a*L**2/(6*E*I)) * (x_p/L - (x_p/L)**3)

                # Zasuki
                alfa_Ap = np.arctan(F * self.gred1_l1 * L  / (6 * E * I * L))
                alfa_Bp = np.arctan(F * self.gred1_l1 * L / (3 * E * I * L))

                # Skupni potek povesov in skupni zasuki
                f_xy = np.concatenate((f1, f2), 0) + f_previs
                alfa_A_skupni_xy = alfa_A + alfa_Ap
                alfa_B_skupni_xy = alfa_B + alfa_Bp

            else:
                f_xy = np.concatenate((f1, f2), 0)
                alfa_A_skupni_xy = alfa_A
                alfa_B_skupni_xy = alfa_B

            # X-Z ravnina
            F = self.Ft1
            # Potek povesov
            x1 = np.arange(0, self.gred1_l1, 0.1)
            f1 = (F * L ** 3 / (6 * E * I)) * ((x1 / L) ** 2 + (x1 / L - 2) * (x1 / L))
            x2 = np.arange(self.gred1_l1, L, 0.1)
            f2 = (F * L ** 3 / (6 * E * I)) * ((x2 / L) ** 2 + (x2 / L - 2) * (x2 / L))
            # Zasuki
            alfa_A = np.arctan(F * self.gred1_l1 * self.gred1_l2 * (L + self.gred1_l2) / (6 * E * I * L))
            alfa_B = np.arctan(F * self.gred1_l1 * self.gred1_l2 * (L + self.gred1_l1) / (6 * E * I * L))

            # Ce upostevamo se previs
            if previs:
                a = previs['a']
                F_p = previs['F']
                x_p = np.arange(0, L, 0.1)
                f_previs = -(F_p * a * L ** 2 / (6 * E * I)) * (x_p / L - (x_p / L) ** 3)

                # Zasuki
                alfa_Ap = np.arctan(F * self.gred1_l1 * L / (6 * E * I * L))
                alfa_Bp = np.arctan(F * self.gred1_l1 * L / (3 * E * I * L))

                # Skupni potek povesov in skupni zasuki
                f_xz = np.concatenate((f1, f2), 0) + f_previs
                alfa_A_skupni_xz = alfa_A + alfa_Ap
                alfa_B_skupni_xz = alfa_B + alfa_Bp

            else:
                f_xz = np.concatenate((f1, f2), 0)
                alfa_A_skupni_xz = alfa_A
                alfa_B_skupni_xz = alfa_B

            # Upostevanje obeh ravnin
            f = (f_xy ** 2 + f_xz ** 2) ** (1 / 2)
            alfa_A_skupni = (alfa_A_skupni_xy ** 2 + alfa_A_skupni_xz ** 2) ** (1 / 2)
            alfa_B_skupni = (alfa_B_skupni_xy ** 2 + alfa_B_skupni_xz ** 2) ** (1 / 2)
            f_pod_zobnikom = f[self.gred1_l1 * 10]
            f_maksimalni = max(abs(f))

            self.gredi[gred]['povesi'] = {'pod zobnikom': f_pod_zobnikom, 'maksimalni': f_maksimalni}
            self.gredi[gred]['zasuki'] = {'A': alfa_A_skupni, 'B': alfa_B_skupni}

        if gred == 3:
            L = self.gred3_l1 + self.gred3_l2

            # X-Y ravnina
            F = self.Fr4
            # Potek povesov
            x1 = np.arange(0, self.gred3_l1, 0.1)
            f1 = (F * L ** 3 / (6 * E * I)) * ((x1 / L) ** 2 + (x1 / L - 2) * (x1 / L))
            x2 = np.arange(self.gred3_l1, L, 0.1)
            f2 = (F * L ** 3 / (6 * E * I)) * ((x2 / L) ** 2 + (x2 / L - 2) * (x2 / L))

            # Zasuki
            alfa_A = np.arctan(F * self.gred3_l1 * self.gred3_l2 * (L + self.gred3_l2) / (6 * E * I * L))
            alfa_B = np.arctan(F * self.gred3_l1 * self.gred3_l2 * (L + self.gred3_l1) / (6 * E * I * L))

            # Ce upostevamo se previs
            if previs:
                a = previs['a']
                F_p = previs['F']
                x_p = np.arange(0, L, 0.1)
                f_previs = -(F_p * a * L ** 2 / (6 * E * I)) * (x_p / L - (x_p / L) ** 3)

                # Zasuki
                alfa_Ap = np.arctan(F * self.gred3_l1 * L / (6 * E * I * L))
                alfa_Bp = np.arctan(F * self.gred3_l1 * L / (3 * E * I * L))

                # Skupni potek povesov in skupni zasuki
                f_xy = np.concatenate((f1, f2), 0) + f_previs
                alfa_A_skupni_xy = alfa_A + alfa_Ap
                alfa_B_skupni_xy = alfa_B + alfa_Bp

            else:
                f_xy = np.concatenate((f1, f2), 0)
                alfa_A_skupni_xy = alfa_A
                alfa_B_skupni_xy = alfa_B

            # X-Z ravnina
            F = self.Ft4
            # Potek povesov
            x1 = np.arange(0, self.gred3_l1, 0.1)
            f1 = (F * L ** 3 / (6 * E * I)) * ((x1 / L) ** 2 + (x1 / L - 2) * (x1 / L))
            x2 = np.arange(self.gred3_l1, L, 0.1)
            f2 = (F * L ** 3 / (6 * E * I)) * ((x2 / L) ** 2 + (x2 / L - 2) * (x2 / L))

            # Zasuki
            alfa_A = np.arctan(F * self.gred3_l1 * self.gred3_l2 * (L + self.gred3_l2) / (6 * E * I * L))
            alfa_B = np.arctan(F * self.gred3_l1 * self.gred3_l2 * (L + self.gred3_l1) / (6 * E * I * L))

            # Ce upostevamo se previs
            if previs:
                a = previs['a']
                F_p = previs['F']
                x_p = np.arange(0, L, 0.1)
                f_previs = -(F_p * a * L ** 2 / (6 * E * I)) * (x_p / L - (x_p / L) ** 3)

                # Zasuki
                alfa_Ap = np.arctan(F * self.gred3_l1 * L / (6 * E * I * L))
                alfa_Bp = np.arctan(F * self.gred3_l1 * L / (3 * E * I * L))

                # Skupni potek povesov in skupni zasuki
                f_xz = np.concatenate((f1, f2), 0) + f_previs
                alfa_A_skupni_xz = alfa_A + alfa_Ap
                alfa_B_skupni_xz = alfa_B + alfa_Bp

            else:
                f_xz = np.concatenate((f1, f2), 0)
                alfa_A_skupni_xz = alfa_A
                alfa_B_skupni_xz = alfa_B

            # Upostevanje obeh ravnin
            f = (f_xy ** 2 + f_xz ** 2) ** (1 / 2)
            alfa_A_skupni = (alfa_A_skupni_xy ** 2 + alfa_A_skupni_xz ** 2) ** (1 / 2)
            alfa_B_skupni = (alfa_B_skupni_xy ** 2 + alfa_B_skupni_xz ** 2) ** (1 / 2)
            f_pod_zobnikom = f[self.gred3_l1 * 10]
            f_maksimalni = max(abs(f))

            self.gredi[gred]['povesi'] = {'pod zobnikom': f_pod_zobnikom, 'maksimalni': f_maksimalni}
            self.gredi[gred]['zasuki'] = {'A': alfa_A_skupni, 'B': alfa_B_skupni}

    # --------Vrednotenje zobnikov--------
    def korenska_nosilnost(self, zveza, Yfa, Ysa, Kfa, Kv, sigmaflim, YRrelT, Yx, Ybeta = 1, Kfb = 1.5, Ka = 1.5, YST = 2, YNT = 1, YdrelT = 1,
                           Sfmin = 1.6):
        """Metoda po standardu DIN 3990 in ISO 6336 izračuna upogibno napetost v korenu zoba ter dopustno upogibno napetost."""

        # Glede na izbrano zvezo (12 ali 34) določimo ustrezne veličine
        if zveza == 12:
            da1 = self.dai()[1]; db1 = self.dfi()[1];
            da2 = self.dai()[2]; db2 = self.dfi()[2];
            a = self.ad12; b = self.b12; m = self.m12;
            Ft = self.Ft1

        if zveza == 34:
            da1 = self.dai()[3]; db1 = self.dfi()[3];
            da2 = self.dai()[4]; db2 = self.dfi()[4];
            a = self.ad23; b = self.b34; m = self.m12;
            Ft = self.Ft3

        # Glede na podane veličine izračunamo ea
        ea = ((da1**2 - db1**2)**(1/2) + (da2**2 - db2**2)**(1/2) - 2*a*np.sin(20 * np.pi/180)) / (2*np.pi*m)
        Ye = 0.25 + 0.75 / ea

        # Izračunamo dejansko korensko napetost
        sigma_f = Ft/(b*m) * Yfa * Ysa * Ye * Ybeta * Kfa * Kfb * Ka * Kv

        # Ter dopustno korensko napetosti+
        sigma_dop = sigmaflim * YST * YNT * YdrelT * YRrelT * Yx / Sfmin

        # In vrednosti dodamo glavnemu slovarju za zobnike
        if zveza == 12:
            self.zobniki[1]["sigma_F"] = sigma_f
            self.zobniki[1]["sigma_FP"] = sigma_dop
            self.zobniki[2]["sigma_F"] = sigma_f
            self.zobniki[2]["sigma_FP"] = sigma_dop
        if zveza == 34:
            self.zobniki[3]["sigma_F"] = sigma_f      
            self.zobniki[3]["sigma_FP"] = sigma_dop   
            self.zobniki[4]["sigma_F"] = sigma_f      
            self.zobniki[4]["sigma_FP"] = sigma_dop   

    def bocna_nosilnost(self, zveza, Zeps, sigmahlim, Kv, ZX, Ka = 1.5, ZL=None, ZV=None, ZR=None, ZNT=1.3, KHb = 1.65, KHa = 1.1, Zw = 1, \
                        SHmin = 1.2, ZH = 2.5, ZE = 190, Zbeta = 1):
        """Metoda po standardu DIN 3990 in ISO 6336 izračuna upogibno napetost v korenu zoba ter dopustno upogibno napetost."""

        # Glede na izbrano zvezo (12 ali 34) določimo ustrezne veličine
        if zveza == 12:
            d1 = self.dai()[1]
            b = self.b12
            Ft = self.Ft1
            u = self.z2/self.z1

        if zveza == 34:
            d1 = self.dai()[3]
            b = self.b34
            Ft = self.Ft3
            u = self.z4 / self.z3

        # Izračunamo dejansko bocno napetosti
        sigmaH = ZH * ZE * Zeps * Zbeta * (KHa * KHb * Ka * Kv * ((u+1)/u) * Ft / (b * d1))**(1/2)

        # Izračunamo dopustno bocno napetost
        if ZL == None:
            sigmaHP = sigmahlim * ZNT * 0.92 * Zw * ZX / SHmin
        else:
            sigmaHP = sigmahlim * ZNT * ZL * ZV * ZR * Zw * ZX / SHmin

        # In dodamo glavnemu slovarju zobnikov
        if zveza == 12:
            self.zobniki[1]["sigma_H"] = sigmaH
            self.zobniki[1]["sigma_HP"] = sigmaHP
            self.zobniki[2]["sigma_H"] = sigmaH
            self.zobniki[2]["sigma_HP"] = sigmaHP
        if zveza == 34:
            self.zobniki[3]["sigma_H"] = sigmaH
            self.zobniki[3]["sigma_HP"] = sigmaHP
            self.zobniki[4]["sigma_H"] = sigmaH
            self.zobniki[4]["sigma_HP"] = sigmaHP


    # --------Izrisi in zapisi končnega poročila--------
    def zapisi_txt(self, pot):
        """Metoda sprejme parameter pot in po klicanih vseh potrebnih funkcijah zapiše končno poročilo celotnega projekta."""

        file = open(pot, 'w+')
        file.write('______________________________________\n')
        file.write('|To je porocilo za Strojne elemente2!|\n')
        file.write('______________________________________\n\n')

        file.write(f'Vhodni podatki so naslednji:\n')
        vhodni_podatki = [self.P1,
                          self.n1,
                          self.n3,
                          self.m12,
                          self.m34,
                          self.z1,
                          self.z3,
                          self.beta1,
                          self.beta2]
        vhodni_podatki_krat = ['P1', 'n1', 'n3', 'm12', 'm34', 'z1', 'z3', 'beta1', 'beta2']
        vhodni_podatki_enote = ['W', 'min^-1', 'min^-1', 'mm', 'mm', ' ', ' ', '°', '°']
        for ime, pod, eno in zip(vhodni_podatki_krat, vhodni_podatki, vhodni_podatki_enote):
            file.write(f'{ime: >5} = {pod:7.1f} {eno}\n')

        file.write(f'\nPrestave:\n-------------------------------------------\n')
        file.write(f'Teoretična skupna prestava isk = {self.isk:.2f}.\n')
        file.write(f'Teoretična prestava i12 = {self.i12:.2f}.\n')
        file.write(f'Teoretična prestava i34 = {self.i34:.2f}.\n')
        file.write('\n\n')

        file.write(f'Število zob:\n-------------------------------------------\n')
        file.write(f'z1 = {self.z1}\n')
        file.write(f'z2 = {self.z2}\n')
        file.write(f'z3 = {self.z3}\n')
        file.write(f'z4 = {self.z4}\n')
        file.write('\n\n')

        file.write(f'Dejanske prestave:\n-------------------------------------------\n')
        file.write(f'i12,dej = {self.i12_dej:.2f}\n')
        file.write(f'i34,dej = {self.i34_dej:.2f}\n')
        file.write(f'is,dej = {self.isk_dej:.2f}\n')
        file.write('\n\n')

        file.write(f'Dejanske vrtilne hitrosti:\n-------------------------------------------\n')
        file.write(f'omega1 = {self.omega1:.2f} rad/s\n')
        file.write(f'omega2 = {self.omega2:.2f} rad/s\n')
        file.write(f'omega3 = {self.omega3:.2f} rad/s\n')
        file.write('\n\n')

        file.write(f'Moč na delovnem stroju:\n-------------------------------------------\n')
        file.write(f'Pds = {self.Pds:.2f} W\n')
        file.write('\n\n')

        file.write(f'Momenti na gredeh:\n-------------------------------------------\n')
        file.write(f'T1 = {self.T1:.2f} Nm\n')
        file.write(f'T2 = {self.T2:.2f} Nm\n')
        file.write(f'T3 = {self.T3:.2f} Nm\n')
        file.write('\n\n')

        file.write(f'Minimalni premeri gredi:\n-------------------------------------------\n')
        file.write(f'd1 = {self.d1():.2f} mm\n')
        file.write(f'd2 = {self.d2():.2f} mm\n')
        file.write(f'd3 = {self.d3():.2f} mm\n')
        file.write('\n\n')

        file.write(f'Geometrija zobnikov:\n-------------------------------------------\n')
        for zobnik in [1,2,3,4]:
            file.write(f'Zobnik {zobnik}:\n')
            file.write(f'kinematski premer d = {self.di()[zobnik]:.1f} mm\n')
            file.write(f'temenski premer da = {self.dai()[zobnik]:.1f} mm\n')
            file.write(f'korenski premer df = {self.dfi()[zobnik]:.1f} mm\n')
            file.write('\n')

        file.write('Medosna razdalja med gredjo 1 in 2:\n')
        file.write(f'a12 = {self.ad12:.1f} mm\n')
        file.write('Medosna razdalja med gredjo 2 in 3:\n')
        file.write(f'a23 = {self.ad23:.1f} mm\n')
        file.write('\n')

        file.write('Širina zobnikov:\n')
        file.write(f'b12 = {self.b12 :.1f} mm\n')
        file.write(f'b34 = {self.b34 :.1f} mm\n\n')

        file.write(f'Statika gredi 2:\n-------------------------------------------\n')
        file.write(f'Ft2 = {self.Ft2:.2f} mm\n')
        file.write(f'Ft3 = {self.Ft3:.2f} mm\n')
        file.write(f'Fr2 = {self.Fr2:.2f} mm\n')
        file.write(f'Fr3 = {self.Fr3:.2f} mm\n\n')

        file.write('X-Y ravnina\n')
        file.write(f'Ay = {self.gred2_Ay:.2f} N\n')
        file.write(f'By = {self.gred2_By:.2f} N\n')
        file.write(f'Mzmax = {self.gred2_Mzmax:.2f} Nm\n')

        file.write('X-Z ravnina\n')
        file.write(f'Az = {self.gred2_Az:.2f} N\n')
        file.write(f'Bz = {self.gred2_Bz:.2f} N\n')
        file.write(f'Mymax = {self.gred2_Mymax:.2f} Nm\n')

        file.write('Skupne obremenitve 2. gredi\n')
        file.write(f'A = {self.gred2_A :.2f} N\n')
        file.write(f'B = {self.gred2_B:.2f} N\n')
        file.write(f'Mmax = {self.gred2_Mmax:.2f} Nm\n\n')

        file.write(f'Ležaji na gredi 2:\n-------------------------------------------\n')
        file.write('Statična nosilnost\n')
        file.write('Ležaj v podpori A:\n')
        file.write(f'P0 = {self.lezaji[2]["P0"]["A"]:.2f} N\n')
        file.write(f'S0 = {self.lezaji[2]["S0"]["A"]:.3f}\n')
        file.write(f'S0min = {self.lezaji[2]["S0min"]["A"]:.1f}\n\n')
        file.write('Ležaj v podpori B:\n')
        file.write(f'P0 = {self.lezaji[2]["P0"]["B"]:.2f} N\n')
        file.write(f'S0 = {self.lezaji[2]["S0"]["B"]:.3f}\n')
        file.write(f'S0min = {self.lezaji[2]["S0min"]["B"]:.1f}\n\n')

        file.write('Imenska doba trajanja\n')
        file.write('Ležaj v podpori A:\n')
        file.write(f'P = {self.lezaji[2]["P"]["A"]:.2f} N\n')
        file.write(f'L10 = {self.lezaji[2]["L10"]["A"]:.3f}\n')
        file.write(f'L10h = {self.lezaji[2]["L10h"]["A"]:.1f} h\n\n')
        file.write('Ležaj v podpori B:\n')
        file.write(f'P = {self.lezaji[2]["P"]["B"]:.2f} N\n')
        file.write(f'L10 = {self.lezaji[2]["L10"]["B"]:.3f}\n')
        file.write(f'L10h = {self.lezaji[2]["L10h"]["B"]:.1f} h\n\n')

        file.write('Razširjen izračun dobe trajanja\n')
        file.write('Ležaj v podpori A:\n')
        file.write(f'Lna = {self.lezaji[2]["Lna"]["A"]:.3f}\n')
        file.write(f'Lna_h = {self.lezaji[2]["Lna_h"]["A"]:.3f} h\n\n')
        file.write('Ležaj v podpori B:\n')
        file.write(f'Lna = {self.lezaji[2]["Lna"]["B"]:.3f}\n')
        file.write(f'Lna_h = {self.lezaji[2]["Lna_h"]["B"]:.3f} h\n\n\n')

        file.write(f'Kritični prerez pod zobnikoma na gredi 2:\n-------------------------------------------\n')
        file.write('Levi zobnik:\n')
        file.write(f'sigma_fmax = {self.gredi[2]["prerezi"]["pod zobnikom"]["L"]["sigma_fmax"]:.2f} MPa\n')
        file.write(f'tau_tmax = {self.gredi[2]["prerezi"]["pod zobnikom"]["L"]["tau_tmax"]:.2f} MPa\n')
        file.write(f'Primerjalna___sigma_p = {self.gredi[2]["prerezi"]["pod zobnikom"]["L"]["sigma_p"]:.2f} MPa\n')
        file.write(f'Dopustna___sigma_dop = {self.gredi[2]["prerezi"]["pod zobnikom"]["L"]["sigma_dop"]:.2f} MPa\n\n')
        file.write('Desni zobnik:\n')
        file.write(f'sigma_fmax = {self.gredi[2]["prerezi"]["pod zobnikom"]["D"]["sigma_fmax"]:.2f} MPa\n')
        file.write(f'tau_tmax = {self.gredi[2]["prerezi"]["pod zobnikom"]["D"]["tau_tmax"]:.2f} MPa\n')
        file.write(f'Primerjalna___sigma_p = {self.gredi[2]["prerezi"]["pod zobnikom"]["D"]["sigma_p"]:.2f} MPa\n')
        file.write(f'Dopustna___sigma_dop = {self.gredi[2]["prerezi"]["pod zobnikom"]["D"]["sigma_dop"]:.2f} MPa\n\n')

        file.write(f'Kritični prerez stopnica na gredi 2:\n-------------------------------------------\n')
        file.write(f'sigma_fmax = {self.gredi[2]["prerezi"]["stopnica"]["sigma_fmax"]:.2f} MPa\n')
        file.write(f'tau_tmax = {self.gredi[2]["prerezi"]["stopnica"]["tau_tmax"]:.2f} MPa\n')
        file.write(f'Primerjalna___sigma_p = {self.gredi[2]["prerezi"]["stopnica"]["sigma_p"]:.2f} MPa\n')
        file.write(f'Dopustna___sigma_dop = {self.gredi[2]["prerezi"]["stopnica"]["sigma_dop"]:.2f} MPa\n\n')

        file.write(f'Povesi in zasuki gredi 2:\n-------------------------------------------\n')
        file.write(f'maksimalni poves fmax = {self.gredi[2]["povesi"]["maksimalni"]:.4f} mm\n')
        file.write(f'poves pod zobnikom 2 = {self.gredi[2]["povesi"]["pod zobnikom"][1]:.4f} mm\n')
        file.write(f'poves pod zobnikom 3 = {self.gredi[2]["povesi"]["pod zobnikom"][2]:.4f} mm\n')
        file.write(f'zasuk v ležaju A = {self.gredi[2]["zasuki"]["A"]:.6f}\n')
        file.write(f'zasuk v ležaju B = {self.gredi[2]["zasuki"]["B"]:.6f}\n\n')

        file.write(f'Korenska nosilnost:\n-------------------------------------------\n')
        file.write(f'Med prvim in drugim zobnikom:\n')
        file.write(f'Upogibna napetost v korenu zoba____ sigma_F = {self.zobniki[1]["sigma_F"]:.2f} MPa\n')
        file.write(f'Dopustna napetost____ sigma_FP = {self.zobniki[1]["sigma_FP"]:.2f} MPa\n')
        file.write(f'Med tretjim in četrtim zobnikom:\n')
        file.write(f'Upogibna napetost v korenu zoba____ sigma_F = {self.zobniki[3]["sigma_F"]:.2f} MPa\n')
        file.write(f'Dopustna napetost____ sigma_FP = {self.zobniki[3]["sigma_FP"]:.2f} MPa\n\n\n')

        file.write(f'Bočna nosilnost:\n-------------------------------------------\n')
        file.write(f'Med prvim in drugim zobnikom:\n')
        file.write(f'sigma_H = {self.zobniki[1]["sigma_H"]:.2f} MPa\n')
        file.write(f'sigma_HP = {self.zobniki[1]["sigma_HP"]:.2f} MPa\n')
        file.write(f'Med tretjim in četrtim zobnikom:\n')
        file.write(f'sigma_H = {self.zobniki[3]["sigma_H"]:.2f} MPa\n')
        file.write(f'sigma_HP = {self.zobniki[3]["sigma_HP"]:.2f} MPa\n')

        file.close()
