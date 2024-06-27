import cv2
import numpy as np
from collections import OrderedDict

class MerkezTracker:
    def __init__(self, maxKaybolan=50):
        self.sonrakiNesneID = 0
        self.nesneler = OrderedDict()
        self.kaybolan = OrderedDict()
        self.renkler = OrderedDict()
        self.maxKaybolan = maxKaybolan

    def kaydet(self, merkez):
        self.nesneler[self.sonrakiNesneID] = {"merkez": merkez, "iz": [merkez]}
        self.kaybolan[self.sonrakiNesneID] = 0
        self.renkler[self.sonrakiNesneID] = self._renk_olustur()
        self.sonrakiNesneID += 1

    def kayitsil(self, nesneID):
        del self.nesneler[nesneID]
        del self.kaybolan[nesneID]
        del self.renkler[nesneID]

    def _renk_olustur(self):
        return tuple(np.random.randint(0, 255, size=3).tolist())

    def guncelle(self, merkezler):
        if len(merkezler) == 0:
            for nesneID in list(self.kaybolan.keys()):
                self.kaybolan[nesneID] += 1
                if self.kaybolan[nesneID] > self.maxKaybolan:
                    self.kayitsil(nesneID)
            return self.nesneler

        if len(self.nesneler) == 0:
            for i in range(len(merkezler)):
                self.kaydet(merkezler[i])
        else:
            nesneIDler = list(self.nesneler.keys())
            nesneMerkezler = np.array([nesne['merkez'] for nesne in self.nesneler.values()])
            D = np.linalg.norm(nesneMerkezler[:, np.newaxis] - merkezler, axis=2)

            satirlar = D.min(axis=1).argsort()
            sutunlar = D.argmin(axis=1)[satirlar]

            kullanilanSatirlar = set()
            kullanilanSutunlar = set()

            for (satir, sutun) in zip(satirlar, sutunlar):
                if satir in kullanilanSatirlar or sutun in kullanilanSutunlar:
                    continue
                nesneID = nesneIDler[satir]
                self.nesneler[nesneID]['merkez'] = merkezler[sutun]
                self.nesneler[nesneID]['iz'].append(merkezler[sutun])
                self.kaybolan[nesneID] = 0
                kullanilanSatirlar.add(satir)
                kullanilanSutunlar.add(sutun)

            kullanilmayanSatirlar = set(range(D.shape[0])).difference(kullanilanSatirlar)
            kullanilmayanSutunlar = set(range(D.shape[1])).difference(kullanilanSutunlar)

            if D.shape[0] >= D.shape[1]:
                for satir in kullanilmayanSatirlar:
                    nesneID = nesneIDler[satir]
                    self.kaybolan[nesneID] += 1
                    if self.kaybolan[nesneID] > self.maxKaybolan:
                        self.kayitsil(nesneID)
            else:
                for sutun in kullanilmayanSutunlar:
                    self.kaydet(merkezler[sutun])

        return self.nesneler

def top_bul(frame, alt_renk, ust_renk):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, alt_renk, ust_renk)
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_pozisyonlari = []

    for kontur in konturlar:
        alan = cv2.contourArea(kontur)
        if alan < 100:
            continue
        moments = cv2.moments(kontur)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            top_pozisyonlari.append((cx, cy))

    return top_pozisyonlari

video_yolu = 'vid_1.avi'
cap = cv2.VideoCapture(video_yolu)
mt = MerkezTracker()
x, y, w, h = 40, 60, 600, 310
carpisma_sayisi = 0
carpisma_ciftleri = set()

alt_renk_kirmizi = np.array([160, 100, 100])
ust_renk_kirmizi = np.array([180, 255, 255])
alt_renk_beyaz = np.array([0, 0, 200])
ust_renk_beyaz = np.array([180, 55, 255])

carpisma_mesafe_esigi = 22
beyaz_top_kirmizi_topa_carpti = False
top_izleri = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[y:y + h, x:x + w]
    kirmizi_top_pozisyonlari = top_bul(roi, alt_renk_kirmizi, ust_renk_kirmizi)
    beyaz_top_pozisyonlari = top_bul(roi, alt_renk_beyaz, ust_renk_beyaz)

    izlenen_nesneler = mt.guncelle(kirmizi_top_pozisyonlari)

    for nesneID, nesne_bilgisi in izlenen_nesneler.items():
        merkez = (nesne_bilgisi['merkez'][0] + x, nesne_bilgisi['merkez'][1] + y)
        renk = mt.renkler[nesneID]
        if nesneID not in top_izleri:
            if len(top_izleri) < 7:
                top_izleri[nesneID] = []
            else:
                continue
        top_izleri[nesneID].append(merkez)

        for iz_noktasi in nesne_bilgisi['iz']:
            iz_noktasi = (iz_noktasi[0] + x, iz_noktasi[1] + y)
            cv2.circle(frame, tuple(map(int, iz_noktasi)), 3, renk, -1)

        cv2.circle(frame, tuple(map(int, merkez)), 6, renk, -1)
        cv2.putText(frame, "ID {}".format(nesneID), (merkez[0] - 10, merkez[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)

    if beyaz_top_pozisyonlari:
        beyaz_top_merkezi = beyaz_top_pozisyonlari[0]

        for kirmizi_top_merkezi in kirmizi_top_pozisyonlari:
            mesafe = np.linalg.norm(np.array(beyaz_top_merkezi) - np.array(kirmizi_top_merkezi))
            if mesafe < carpisma_mesafe_esigi:
                beyaz_top_kirmizi_topa_carpti = True
                break

    if beyaz_top_kirmizi_topa_carpti:
        mevcut_cerceve_carpismalar = set()
        nesneIDler = list(izlenen_nesneler.keys())
        for i in range(len(nesneIDler)):
            for j in range(i + 1, len(nesneIDler)):
                id1, id2 = nesneIDler[i], nesneIDler[j]
                mesafe = np.linalg.norm(
                    np.array(izlenen_nesneler[id1]['merkez']) - np.array(izlenen_nesneler[id2]['merkez']))
                if mesafe < carpisma_mesafe_esigi:
                    carpisma_cifti = tuple(sorted((id1, id2)))
                    if carpisma_cifti not in carpisma_ciftleri:
                        mevcut_cerceve_carpismalar.add(carpisma_cifti)
                        carpisma_ciftleri.add(carpisma_cifti)
                        carpisma_sayisi += 1

        for id1, id2 in mevcut_cerceve_carpismalar:
            pt1 = tuple(map(int, izlenen_nesneler[id1]['merkez']))
            pt2 = tuple(map(int, izlenen_nesneler[id2]['merkez']))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    cv2.putText(frame, "Carpisma: {}".format(carpisma_sayisi), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Bilardo Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for nesneID in list(top_izleri.keys())[:7]:
    if nesneID in mt.renkler:
        iz_frame = np.zeros((h, w, 3), dtype=np.uint8)
        renk = mt.renkler[nesneID]
        for nokta in top_izleri[nesneID]:
            cv2.circle(iz_frame, (nokta[0] - x, nokta[1] - y), 3, renk, -1)
        cv2.putText(iz_frame, "ID {}".format(nesneID), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)
        cv2.imwrite(f"top_{nesneID}_iz.png", iz_frame)

