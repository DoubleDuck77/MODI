import lightkurve as lk 
#scam totalny, potrzebuje jeszcze numpy o wersji <2.0, oktopus i sto tysięcy jakichś dziwactw totalnych, j a to robię w venv
import matplotlib.pyplot as plt

#tutaj szukasz danych tessu i spółki
sr = lk.search_lightcurve("TOI-181", mission="TESS")
print(sr)

lc = sr[0].download() 
lc = lc.normalize()    

lc.plot()
plt.savefig("toi181_lightcurve.png")
