# ECG Baseline Removal

This repository contains code for removing the baseline wander from ECG (Electrocardiogram) data. The baseline wander refers to the low-frequency noise and drift that can be present in ECG signals.
The code applies filtering and interpolation techniques to remove the baseline wander and obtain a cleaner ECG signal.

## Dependencies

The code requires the following dependencies:

- NumPy
- SciPy
- biosppy
- Matplotlib

You can install the dependencies using `pip` with the following command:
```commandline
pip install numpy scipy biosppy matplotlib

```

## Parameters

- ecg_data: NumPy vector containing the ECG data of a single lead.
- sample_rate: Sampling rate of the ECG data.
- mode: The mode of interpolation to be used ('linear' or 'cubic_spline'). Default is 'linear'.
- number_of_neighbors: The number of samples to consider before and after the center index. Default is 10.
- seconds_to_remove: The number of seconds to trim from the start and end of the ECG data. Default is 1.
- verbose: Boolean flag indicating whether to display plots for visualization. Default is False.

## License
This code is provided under the MIT License. Feel free to use and modify it according to your requirements.