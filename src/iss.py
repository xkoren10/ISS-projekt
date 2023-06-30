# ISS Projekt 2021/22, Autor: Matej Koreň

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import spectrogram, freqz, tf2zpk, lfilter
from timeit import default_timer as timer
from datetime import timedelta


################
# Vektorová dft
################
def dft(array):
    x = np.asarray(array, dtype=float)
    N = len(array)
    n = np.arange(2)
    k = n[:, np.newaxis]
    dft_mat = np.exp(-2j * np.pi * n * k / 2)
    ft_x = np.matmul(dft_mat, x.reshape((2, -1)))

    while ft_x.shape[0] < N:
        ft_x_even = ft_x[:, :ft_x.shape[1] // 2]
        ft_x_odd = ft_x[:, ft_x.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(ft_x.shape[0]) / ft_x.shape[0])[:, np.newaxis]
        ft_x = np.vstack([ft_x_even + factor * ft_x_odd, ft_x_even - factor * ft_x_odd])

    return ft_x.ravel()


################
# Načítanie
################

s, fs = sf.read('audio/xkoren10.wav')
t = np.arange(s.size) / fs
duration = len(s)/fs

print(f'Vzorkovacia frekvencia: {fs}')
print(f'Dĺžka vo vzorkoch: {len(s)}')
print(f'Dĺžka [s]: {duration}')
print(f'Maximum: {max(s)}')
print(f'Minimum: {min(s)}')


plt.figure(figsize=(6, 3))
plt.plot(t, s)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')
plt.tight_layout()
plt.show()


################
# Centralizácia
################

for k in range(0, s.size):
    s[k] = s[k] - np.mean(s)

plt.figure(figsize=(6, 3))
plt.plot(s)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Ustrednenie')
plt.tight_layout()
plt.show()


#####################
# Rozdelenie na rámce
#####################

overlap_size = 512
window_size = 1024
number_of_frames = len(s)//1024

frames = np.ndarray((number_of_frames, window_size))

for k in range(0, number_of_frames):
    for i in range(0, window_size):
        if (k * overlap_size + i) < len(s):
            frames[k][i] = s[k * overlap_size + i]
        else:
            frames[k][i] = 0

main_plot = plt.figure(figsize=(9, 5))
main_plot.add_subplot(1, 1, 1)
plt.xlim([0, 320])
plt.plot(frames[24])
locals_f, labels = plt.xticks()
labels = [float(item) * 0.0625 for item in locals_f]
plt.xticks(locals_f, labels)
plt.xlabel('Čas[ms]')
plt.ylabel('Amplitúda')
plt.title('Vlnová funkcia v okne č.24')
plt.show()

# Pekné rámce - 2,4,10,24
nice_frame = frames[24]

start = timer()
dft_frame = dft(nice_frame)
end = timer()


proximity = np.allclose(dft_frame, np.fft.fft(nice_frame))
print("Podobnosť s fft : " + str(proximity) + ", Čas : " + str(timedelta(seconds=end-start)))


spec_axis = np.arange(0, fs, fs/len(dft_frame))
plt.plot(spec_axis[:len(dft_frame) // 2], np.abs(dft_frame[:len(dft_frame) // 2]))
plt.xlabel('Frekvencia [Hz]')
plt.title(f'Segment po DFT')
plt.draw()
plt.show()

#####################
# Spektrogram
#####################

f, t, sgr = spectrogram(s, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(abs(sgr+1e-20)**2)
plt.pcolormesh(t, f, sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvencia [Hz]')
plt.title('Logaritmický výkonový spektrogram pôvodného signálu')
plt.draw()
plt.tight_layout()
plt.show()


#####################
# Rušivé frekvencie
#####################

dist_freqs = [0, 0, 0, 0]
dist_freqs[0] = (spec_axis[np.argmax(np.abs(dft_frame[:100]))])
dist_freqs[1] = dist_freqs[0] * 2
dist_freqs[2] = 2515.5
dist_freqs[3] = 3359.3

print("Rušivé frekvencie : " + str(dist_freqs))

u = np.arange(0, len(s), 1)
y1 = np.cos(dist_freqs[0] / fs * 2 * np.pi * u)
y2 = np.cos(dist_freqs[1] / fs * 2 * np.pi * u)
y3 = np.cos(dist_freqs[2] / fs * 2 * np.pi * u)
y4 = np.cos(dist_freqs[3] / fs * 2 * np.pi * u)
y = y1+y2+y3+y4

sf.write('audio/4cos.wav', y/69, fs)

f, t, sgr = spectrogram(y, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(abs(sgr+1e-20)**2)
plt.pcolormesh(t, f, sgr_log)
plt.xlabel('Čas [s]')
plt.ylabel('Frekvencie [Hz]')
plt.title('Logaritmický výkonový spektrogram 4cos.wav')
plt.draw()
plt.tight_layout()
plt.show()


#####################
# Návrh filtru
#####################

wp = 50
ws = 5
filter_a = filter_b = 1

for f in dist_freqs:
    N, wn = signal.buttord([f-wp, f+wp], [f-ws, f+ws], 3, 40, False, fs)
    b, a = signal.butter(N, wn, 'bandstop', False, 'ba', fs)

# Spojenie zádrží do filtrov
    filter_a = np.convolve(filter_a, a)
    filter_b = np.convolve(filter_b, b)

print("Koeficient A :" + str(filter_a))
print("Koeficient B :" + str(filter_b))


#####################
# Impulzná odozva
#####################

imp = [1, *np.zeros(31)]
response = signal.lfilter(b, a, imp)

plt.title('Impulzná odozva filtru')
plt.stem(np.arange(32), response, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.grid(alpha=0.5, linestyle='-')
plt.tight_layout()
plt.show()

#############
# Nuly a póly
#############

z, p, k = tf2zpk(filter_b, filter_a)
w, H = freqz(filter_b, filter_a)

###########
# Stabilita
###########

is_stable = (p.size == 0) or np.all(np.abs(p) < 1)
print('Filter{} je stabilný.'.format('' if is_stable else ' nie'))

#####################
# Jednotková kružnica
#####################

plt.figure(figsize=(4, 3.5))
ang = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(ang), np.sin(ang))

plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

plt.gca().set_xlabel('Reálna zložka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginárna zložka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

#######
# Grafy
#######

_, ax = plt.subplots(1, 2, figsize=(10, 3))

ax[0].plot(w / 2 / np.pi * fs, np.abs(H))
ax[0].set_xlabel('Frekvencia [Hz]')
ax[0].set_title('Modul frekvenčnej charakteristiky $|H(e^{j\omega})|$')

ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
ax[1].set_xlabel('Frekvence [Hz]')
ax[1].set_title('Argument frekvenčnej charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()

###########
# Filtrácia
###########

filtered_data = lfilter(filter_b, filter_a, s)
sf.write('audio/clean_bandstop.wav',  np.real(filtered_data), fs)

#######################
# Grafy čistého signálu
#######################

s, fs = sf.read('audio/clean_bandstop.wav')
t = np.arange(s.size) / fs

plt.figure(figsize=(6, 3))
plt.plot(t, s)

plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Vyčistený zvukový signál')

plt.tight_layout()
plt.show()

f, t, sgr = spectrogram(s, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(abs(sgr+1e-20)**2)
plt.pcolormesh(t, f, sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvencia [Hz]')
plt.title('Logaritmický výkonový spektrogram vyčisteného signálu')
plt.draw()
plt.tight_layout()
plt.show()
