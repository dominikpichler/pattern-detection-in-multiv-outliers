import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from scipy.fft import fft, fftfreq
from numpy.fft import fft, ifft
import math
import numpy as np
import matplotlib.pyplot as plt

def apply_fourier_transformation(rawData:pd.DataFrame, label) -> pd.DataFrame:


    df_rawData = rawData.copy()
    df_rawData.columns = ['x', 'y']
    df_rawData = df_rawData.sort_values('x')


    x = df_rawData.iloc[:, 0]
    y = df_rawData.iloc[:, 1]



    lst = []
    for i in range(len(x)):
        lst.append(i)


    n_polyDeg = 2
    accurateModel_found = False


    while(accurateModel_found == False):

        mymodel = np.poly1d(np.polyfit(x, y,n_polyDeg ))
        r_2_score = r2_score(y, mymodel(x))
        mape = mean_absolute_percentage_error(y,mymodel(x))

        if(r_2_score > 0.95 and mape < 0.1):
            accurateModel_found = True

        if n_polyDeg > 10:

            print(" ***** Caution @ [FOURIER]: No sufficiently approximating model for the observations found ***** ")
            return pd.DataFrame()


            break

        n_polyDeg += 1

    print("----- Finished Modelling -----")
    print("Sufficiently approximating model found")
    print("-- Parameters of Model --")
    print("Mape: " + str(mape))
    print("R_2: " + str(r_2_score))
    print("f(x) is a polynom of order " + str(n_polyDeg))
    print("The coeffs for the polynom are: " + str(mymodel.c))






    print("-------------------------------")

    print("----- Starting the Fourier Transformation -----")


    fig = plt.figure(figsize=(10, 7))

    plt.scatter(x,mymodel(x),label = "Original Observations")
    plt.plot(x,y, "r-", label = "Model")
    plt.savefig("results_plots/Modelling_for_DFT" + label + ".png")
    plt.legend()
    plt.ylabel("Dim 1")
    plt.xlabel("Dim ")
    plt.title("Model Fit")

    fig.set_size_inches(w=6.5, h=3.5)
    plt.savefig("results_plots/Modelling_for_DFT" + label + ".pgf")
    plt.show()


    # ======= FOURIER
    #TODO: Generalize

    sr = 600
    ts = 1.0 / sr
    t = np.arange(min(x), max(x), ts)
    x = mymodel(t)

    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N / sr
    freq = n / T

    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(X)*ts, 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 10)
    plt.title("Frequency Domain")

    plt.subplot(122)
    plt.plot(t, ifft(X), 'r')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    labelFourier_plot = "results_plots/" + "FourierSeries_of_" + label + ".png"
    plt.savefig(labelFourier_plot)
    plt.title("Space Domain")

    plt.show()

    df_coeffs_FourierSeries = pd.DataFrame({'Frequency': freq, 'FFT_Amplitude': np.abs(X)*ts},
                                           columns=['Frequency', 'FFT_Amplitude'])




    #---------- CUT-OFF / Qualitiy Testing:

    #Nyquist frequency cut off
    n_cut = len(df_coeffs_FourierSeries)/2
    # pick relevant ones
    df_coeffs_FourierSeries = df_coeffs_FourierSeries.iloc[:-int(n_cut), :]

    df_coeffs_F_sorted = df_coeffs_FourierSeries.sort_values("FFT_Amplitude",ascending=False)

    i = 0
    n_FComp = 3 #TODO: define automatic cut off! -> R_2*(1-MAPE)?
    y_FourierSeries = 0

    list_FSComponents = []
    #df_rawData["Cluster"] = n_FComp

    plt.figure()
    plt.plot(t,x,label= "Original Observations")
    df_fourierClustered = df_rawData.copy()
    df_fourierClustered["Cluster"] = n_FComp


    while i < n_FComp:
        list_xGroupMemebers = []

        comp_i = df_coeffs_F_sorted.iloc[i, 1] * np.sin(2 * np.pi * df_coeffs_F_sorted.iloc[i, 0] * t)
        if df_coeffs_F_sorted.iloc[i, 0] == 0:
            comp_i += df_coeffs_F_sorted.iloc[i, 1]
            i += 1
            continue

        else:
            #binary one
            y_max = max(comp_i)
            j = 0

            #find local maximas
            while j < len(comp_i):
                if math.isclose(comp_i[j],max(comp_i), rel_tol=y_max*0.05) :
                    #df_rawData.loc[j,"Cluster"] = i
                    list_xGroupMemebers.append(t[j])
                j += 1
                y_FourierSeries += comp_i

            li = np.linspace(5, 5, len(list_xGroupMemebers))
            clusterName = "Cluster_" + str(i)
            plt.scatter(list_xGroupMemebers, li,label=clusterName)
            componentenName = "Component_" + str(i)
            plt.plot(t, 2 * comp_i, label=componentenName)


            # Update in Clustering Version


            m = 0
            while m < len(df_fourierClustered):

                for element in list_xGroupMemebers:

                    if math.isclose(df_fourierClustered.loc[m,"x"],element, rel_tol=abs(element*0.001)):
                        df_fourierClustered.loc[m, "Cluster"] = i
                        #print( str(m) + "Element: " + str(element) + " Is close! to " + str(df_fourierClustered.loc[m,"x"]) )


                m += 1

            i += 1



    plt.ylabel('Dim 1')
    plt.xlabel('Dim 2')
    plt.legend()
    plt.title("Fourier Clustering")
    plt.show()




    # ====== Plot clustered result:
    fourier_clusters  = df_fourierClustered['Cluster'].unique()


    plt.figure()
    for cluster in fourier_clusters:
        df_tmpPlot  = df_fourierClustered[df_fourierClustered["Cluster"] == cluster]
        plt.scatter(df_tmpPlot["x"],df_tmpPlot["y"],label = ("Cluster " + str(cluster)))
    plt.legend()
    plt.ylabel("Dim 1")
    plt.xlabel("Dim 2")
    plt.title("Fourier Clustering")
    plt.show()




    #plt.plot(t,2*y_FourierSeries) #TODO: warum faktor 2?


    label_Fourier_Result_data = "results_data/" + "fourierSeries_for_" + label + ".csv"
    label_FourierClustering = "results_data/" + "fourierClustering_for_" + label + ".csv"
    df_coeffs_FourierSeries.to_csv(label_Fourier_Result_data)
    df_fourierClustered.to_csv(label_FourierClustering)

    print("Successfully calculated the Fourier Transformation")
    print("Results will be stored in 'results_data/fourierResults.csv' ")
    print("-------------------------------")




    return df_coeffs_FourierSeries
