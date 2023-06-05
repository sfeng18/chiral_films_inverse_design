Package for QSSAR of chiral functional film
version 1.0

Description
    This package contains the essensial data and codes for establishing the quantitative structure-spectrum-activity relationship (QSSAR) of chiral functional film, including full processes of screening parameters, forward prediction and inverse design.


Requirements
    To run the codes, you need a Python 3 environment with the following packages installed:
    - numpy
    - scipy
    - matplotlib
    - ujson
    - zlib
    - chardet
    - msgpack-python
    - psutil
    - scikit-learn
    - pytorch

Installation
    The main python scripts can be run directly in Python 3 environment, without installation.

Documenation
    Content
        │  Readme.txt
        │
        ├─Data
        │      dye_ABS.msgz
        │      dye_prop.msgz
        │      sdb.msgz
        │      train_idx.npy
        │      uvdb.msgz
        │      uvdb_trn.msgz
        │
        ├─Forward_Prediction
        │      20_gabs_outfile.csv
        │      Forward_network.py
        │      forward_predicting_network.py
        │      model evaluation.txt
        │      net_predict_ABS_all_121_dye_test_250.pkl
        │      new_spectrum.py
        │      paras.py
        │      random_20gabs.csv
        │      spectrum.py
        │
        ├─Inverse_Design_1_Laser
        │  │  cgan-all-445+520+634.py
        │  │  net_newall_445+520+634_200_5.27.pkl
        │  │  new_simple_net_Full_3p.py
        │  │  spectrum.py
        │  │
        │  ├─445+520
        │  │      445nm+520nm.csv
        │  │      netG_all_445+520.pkl
        │  │      netG_all_445+520_0.pkl
        │  │
        │  ├─445+634
        │  │      445nm+634nm.csv
        │  │      netG_all_445+634_g3.28_20_bx.pkl
        │  │
        │  └─520+634
        │          520nm+634nm.csv
        │          netG_all_520+634_g3.06_5_4g.pkl
        │
        ├─Inverse_Design_2_Fluorescence
        │  │  cgan-test-ABS.py
        │  │  paras.py
        │  │  pick_paras.py
        │  │  spectrum.py
        │  │  testG.py
        │  │
        │  ├─440
        │  │      3D_G_PROP_1.8_440.png
        │  │      cgan_test_result_ABS_440.csv
        │  │      cgan_test_result_PROP_440.csv
        │  │      final_result_PROP_440.csv
        │  │      G_PROP_1.8_440.png
        │  │      netG_ABS_440_1.8.pkl
        │  │      netG_PROP_440_1.8.pkl
        │  │      net_predict_ABS_440.pkl
        │  │      net_predict_PROP_440.pkl
        │  │
        │  ├─467
        │  │      cgan_test_result_ABS_467_1.72.csv
        │  │      cgan_test_result_ABS_467_1.78.csv
        │  │      cgan_test_result_PROP_467.csv
        │  │      final_result_PROP_467.csv
        │  │      G_PROP_1.75_467.png
        │  │      G_PROP_1.7_467.png
        │  │      netG_ABS_467_1.72.pkl
        │  │      netG_PROP_467_1.7.pkl
        │  │      net_predict_ABS_467.pkl
        │  │      net_predict_PROP_467.pkl
        │  │
        │  ├─506
        │  │      cgan_test_result_ABS_506_-1.9.csv
        │  │      cgan_test_result_PROP_506.csv
        │  │      final_result_PROP_506.csv
        │  │      G_PROP_-1.89_506_1.png
        │  │      netG_PROP_506_-1.89.pkl
        │  │      net_predict_ABS_506.pkl
        │  │      net_predict_PROP_506.pkl
        │  │
        │  ├─531
        │  │      cgan_test_result_ABS_531_-1.944.csv
        │  │      cgan_test_result_PROP_531.csv
        │  │      final_result_PROP_531.csv
        │  │      G_PROP_-1.95_531.png
        │  │      netG_ABS_531_-1.944.pkl
        │  │      netG_PROP_531_-1.95.pkl
        │  │      net_predict_ABS_531.pkl
        │  │      net_predict_PROP_531.pkl
        │  │
        │  ├─564
        │  │      cgan_test_result_PROP_564_1.csv
        │  │      final_result_PROP_564.csv
        │  │      G_PROP_-1.6_564.png
        │  │      netG_PROP_564_-1.6.pkl
        │  │      net_predict_PROP_564.pkl
        │  │
        │  ├─606
        │  │      cgan_test_result_ABS_606_2.csv
        │  │      cgan_test_result_PROP_606.csv
        │  │      final_result_PROP_606.csv
        │  │      G_PROP_2_606.png
        │  │      netG_ABS_606_2.pkl
        │  │      netG_PROP_606_2.pkl
        │  │      net_predict_ABS_606.pkl
        │  │      net_predict_PROP_606.pkl
        │  │
        │  ├─628
        │  │      cgan_test_result_ABS_628_-1.62.csv
        │  │      cgan_test_result_PROP_628.csv
        │  │      final_result_PROP_628.csv
        │  │      G_PROP_-1.6_628.png
        │  │      netG_ABS_628_-1.62.pkl
        │  │      net_predict_ABS_628.pkl
        │  │      net_predict_PROP_628.pkl
        │  │
        │  └─666
        │          cgan_test_result_ABS_666_1.6.csv
        │          cgan_test_result_PROP_666.csv
        │          final_result_PROP_666.csv
        │          G_PROP_1.6_666.png
        │          netG_ABS_666_1.6.pkl
        │          netG_PROP_666_1.6.pkl
        │          net_predict_ABS_666.pkl
        │          net_predict_PROP_666.pkl
        │
        └─Screening
                cluster-screen.py
                cluster-screen_trn.py
                uv.py

    Functionality of files in each folder
        Data
            This folder contains the database used by other scripts.
            Files:
                dye_ABS.msgz            Absorption spectra of dyes (in solution)
                dye_prop.msgz           Absorption spectra of dyes (in film)
                sdb.msgz                CD spectra of films
                train_idx.npy           The final training set used to train the forward prediction model
                uvdb.msgz               Transmission spectra of dyed films
                uvdb_trn.msgz           Transmission spectra of transparent films

        Screening
            Scripts for clustering and screening. The outputs are the 1st and 2nd round screening results.
            Files:
                uv.py                   Library file
                cluster-screen.py       Python script for screening the parameters of dyed films
                cluster-screen_trn.py   Python script for screening the parameters of transparent films

            Example:
		>cd Screening
                >python ./cluster-screen.py

        Forward_Prediction
            Scripts and expectd results of forword prediction.
            Files:
                new_spectrum.py                             Library file
                forward_predicting_network.py               Script for training the forward network model
                model evaluation.txt                        Performance of forward network model

                paras.py                                    Parameters for model testing
                spectrum.py                                 Library file
                Forward_network.py                          Script for testing the accuracy of forward network model
                net_predict_ABS_all_121_dye_test_250.pkl    Well trained full-spectrum forward prediction network
                random_20gabs.csv                           Several groups of parameters randomly selected as input data
                20_gabs_outfile.csv                         Expected output data

            Notes:
                "Random 20gabs.csv" is several groups of parameters randomly selected as input data. The data in each row of the file represents a different parameter group, and the 
                data in each column represents a different feature (the first column is "dye", the second column is "thickness", the third column is "stretch degree", and the fourth 
                column is "grayscale").

                "20_gabs_jg_20230228.csv" is the output data, which corresponds to the result obtained after the input data is predicted by the neural network, and the nth line of data
                is the result of the nth line of the input data.
                
            Example:
		>cd Forward_Prediction
                >python ./forward_predicting_network.py   ## This command trains the model and output a model named 'my_net.pkl'
                >python ./Forward_network.py              ## This command tests the model named 'net_predict_ABS_all_121_dye_test_250.pkl'

        Inverse_Design_1_Laser
            Scripts and expectd results of inverse design for the laser-switch application.
            Files:
                spectrum.py                                 Library file
                cgan-all-447+520+634.py                     The reverse network to find the largest G value difference between two wavelengths in 445nm, 520nm and 634nm
                net_newall_447+520+634_200_5.27.pkl         Well trained forward prediction network
                new_simple_net_Full_3p.py                   Training a forward prediction network whose outputs are the g values of 445nm, 520nm, and 634nm wavelengths

            Notes:
                Three folders '447+520', '447+634' and '520+634' includes the corresponding parameters and the saved reverse network model for the largest G difference of two wavelengths.
                
            Example:
		>cd Inverse_Design_1_Laser
                >python ./new_simple_net_Full_3p.py  ## This command trains a 3-point prediction model
                >python ./cgan-all-447+520+634.py    ## This command trains the model for inverse design


        Inverse_Design_2_Fluorescence
            Scripts and expectd results of inverse design for the chiral fluorescence application.
            Files:
                spectrum.py                                 Library file
                paras.py                                    Parameters for model testing
                cgan-test-ABS.py                            The reverse network in this application
                pick_paras.py                               Program to remove duplicate/redundant data in the output
                testG.py                                    Program to plot output results as graphs

            Notes:
                Each subfolder includes the corresponding forward and reverse network model for target wavelengths.
