% -------------------------------------------------------------------------
% COMPREHENSIVE ECG SIGNAL ANALYSIS WITH ML CLASSIFICATION
% -------------------------------------------------------------------------
% MIT-BIH Arrhythmia Database - Record 208
% Updated Version: Uses expert annotations for R-peaks instead of findpeaks
% Includes: Preprocessing, R-peak detection, beat segmentation, 
%           heart rate analysis (overall + windowed), HRV metrics,
%           and machine learning classification (DT, SVM, NN)
% -------------------------------------------------------------------------

clear all;
close all;
clc;

fprintf('---------------------------------------\n');
fprintf('ECG ANALYSIS WITH ML CLASSIFICATION\n');
fprintf('---------------------------------------\n\n');

% -------------------------------------------------------------------------
% PART 1: DATA LOADING AND PREPROCESSING
% -------------------------------------------------------------------------

fprintf('PART 1: DATA LOADING AND PREPROCESSING\n');
fprintf('---------------------------------------\n');

% Load ECG signal
fprintf('Loading ECG data (Record 208)...\n');
[signal, fs, tm] = rdsamp('208', 1);
fs = 360;

fprintf('  Sampling frequency: %d Hz\n', fs);
fprintf('  Time between samples (delta-t): %.2f ms\n', 1000/fs);
fprintf('  Total samples: %d\n', length(signal));
fprintf('  Recording duration: %.1f minutes\n\n', length(signal)/(fs*60));

% Extract MLII lead (Modified Limb Lead II)
% This lead provides clear P waves, QRS complexes, and T waves
% It's the standard lead for rhythm analysis with easily identifiable R-peaks
raw_signal = signal(:, 1);
fprintf('Signal length: %d samples\n', length(raw_signal));

% Calculate recording duration
recording_duration_sec = length(raw_signal) / fs;
recording_duration_min = recording_duration_sec / 60;

% Plot raw ECG signal (10 second window)
figure('Name', 'Raw ECG Data');
plot(tm(1:3600), raw_signal(1:3600), "g");
title('Raw ECG Signal (Record 208)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
legend('Raw Data');
grid on;

% -------------------------------------------------------------------------
% Frequency Analysis of Raw Signal (FFT)
% -------------------------------------------------------------------------

fprintf('\nPerforming frequency analysis...\n');

N = length(raw_signal);
Y_raw = fft(raw_signal);
P_raw = abs(Y_raw / N);
P1_raw = P_raw(1:floor(N/2)+1);
P1_raw(2:end-1) = 2*P1_raw(2:end-1);
f = fs * (0:(N/2)) / N;

fprintf('  Frequency resolution: %.4f Hz\n', fs/N);
fprintf('  Maximum frequency (Nyquist): %d Hz\n\n', fs/2);

% Plot frequency spectrum
figure('Name', 'FFT of Raw Signal');
subplot(2,1,1);
plot(f, P1_raw, 'c');
title('Frequency Spectrum of Raw ECG Signal (Full Range)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 fs/2]);
grid on;

subplot(2,1,2);
plot(f, P1_raw, 'c');
title('Frequency Spectrum of Raw ECG Signal (Zoomed: 0-100 Hz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 100]);
grid on;

% -------------------------------------------------------------------------
% Remove DC Offset (Mean Removal)
% -------------------------------------------------------------------------

fprintf('Removing DC offset...\n');
signal_mean = mean(raw_signal);
signal_zero_mean = raw_signal - signal_mean;
fprintf('  Mean of raw signal (DC offset): %.4f mV\n', signal_mean);
fprintf('  New mean after removal: %.4e mV (≈ 0)\n\n', mean(signal_zero_mean));

% Plot DC removal
figure('Name', 'DC Offset Removal');
subplot(2,1,1);
plot(tm(1:3600), raw_signal(1:3600), 'Color', '#ff00ff');
hold on;
yline(signal_mean, 'r--', 'LineWidth', 2);
title('Raw Signal with Mean (Red Dashed Line)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
legend('Raw Signal', sprintf('Mean = %.3f mV', signal_mean));
grid on;

subplot(2,1,2);
plot(tm(1:3600), signal_zero_mean(1:3600), 'g');
hold on;
yline(0, 'r--', 'LineWidth', 2);
title('Signal After Mean Removal (Zero-Centered)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
legend('Zero-Mean Signal', 'Zero Line');
grid on;

% -------------------------------------------------------------------------
% Baseline Drift Removal (Highpass Filter)
% -------------------------------------------------------------------------

fprintf('Applying highpass filter for baseline drift removal...\n');
filter_order = 4;
cutoff_hp = 0.5;
Wn_hp = cutoff_hp / (fs/2);

fprintf('  Filter order: %d\n', filter_order);
fprintf('  Cutoff frequency: %.1f Hz\n', cutoff_hp);
fprintf('  Normalized cutoff (Wn): %.6f\n\n', Wn_hp);

[b_hp, a_hp] = butter(filter_order, Wn_hp, 'high');
signal_hp_filtered = filtfilt(b_hp, a_hp, signal_zero_mean);

% Plot highpass filter effect
figure('Name', 'Baseline Drift Removal');
samples_10sec = 3600;

subplot(2,1,1);
plot(tm(1:samples_10sec), signal_zero_mean(1:samples_10sec), 'Color', '#ff00ff');
title('Before Highpass Filter (Notice Baseline Wander)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
grid on;

subplot(2,1,2);
plot(tm(1:samples_10sec), signal_hp_filtered(1:samples_10sec), 'g');
title('After Highpass Filter (Baseline Drift Removed)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
grid on;

% Visualize filter frequency response
figure('Name', 'Highpass Filter Frequency Response');
[H_hp, W_hp] = freqz(b_hp, a_hp, 1024, fs);

subplot(2,1,1);
plot(W_hp, abs(H_hp), 'c', 'LineWidth', 1.5);
title('Highpass Filter - Magnitude Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 10]);
xline(cutoff_hp, 'r--', 'Cutoff', 'LineWidth', 1.5);
grid on;

subplot(2,1,2);
plot(W_hp, 20*log10(abs(H_hp)), 'c', 'LineWidth', 1.5);
title('Highpass Filter - Magnitude Response (dB)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
xlim([0 10]);
ylim([-60 5]);
xline(cutoff_hp, 'r--', 'Cutoff', 'LineWidth', 1.5);
yline(-3, 'k--', '-3 dB point', 'LineWidth', 1);
grid on;

% -------------------------------------------------------------------------
% Notch Filter (Powerline Interference Removal)
% -------------------------------------------------------------------------

fprintf('Applying notch filter for powerline interference removal...\n');
notch_freq = 60;
Q = 35;
Wo = notch_freq / (fs/2);
BW = Wo / Q;

fprintf('  Notch frequency: %d Hz\n', notch_freq);
fprintf('  Quality factor (Q): %d\n', Q);
fprintf('  Bandwidth: %.2f Hz\n\n', notch_freq/Q);

[b_notch, a_notch] = iirnotch(Wo, BW);
signal_notch_filtered = filtfilt(b_notch, a_notch, signal_hp_filtered);

% Final preprocessed signal
preprocessed_signal = signal_notch_filtered;

% Visualize notch filter frequency response
figure('Name', 'Notch Filter Frequency Response');
[H_notch, W_notch] = freqz(b_notch, a_notch, 4096, fs);

subplot(2,1,1);
plot(W_notch, abs(H_notch), 'c', 'LineWidth', 1.5);
title('Notch Filter - Magnitude Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([50 70]);
xline(notch_freq, 'r--', sprintf('%d Hz', notch_freq), 'LineWidth', 1.5);
grid on;

subplot(2,1,2);
plot(W_notch, 20*log10(abs(H_notch)), 'c', 'LineWidth', 1.5);
title('Notch Filter - Magnitude Response (dB)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
xlim([50 70]);
xline(notch_freq, 'r--', sprintf('%d Hz', notch_freq), 'LineWidth', 1.5);
grid on;

% Compare raw vs preprocessed signal
figure('Name', 'Signal Comparison');
samples_10s = 3600;

subplot(2,1,1);
plot(tm(1:samples_10s), raw_signal(1:samples_10s), 'Color', '#ff00ff');
title('Raw ECG Signal');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
grid on;

subplot(2,1,2);
plot(tm(1:samples_10s), preprocessed_signal(1:samples_10s), 'g');
title('Preprocessed ECG Signal (Clean)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
grid on;

fprintf('Preprocessing complete!\n\n');

figure('Name', 'Baseline Comparison');
samples_30s = 30 * fs;
plot(tm(1:samples_30s), raw_signal(1:samples_30s), 'Color', '#ff00ff'); hold on;
plot(tm(1:samples_30s), preprocessed_signal(1:samples_30s), 'c');
yline(0, 'w--');
title('Raw vs. Preprocessed ECG (30 s)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
legend('Raw', 'Preprocessed', 'Zero line');
grid on;

% -------------------------------------------------------------------------
% PART 2: R-PEAK DETECTION (USING EXPERT ANNOTATIONS)
% -------------------------------------------------------------------------

fprintf('PART 2: R-PEAK DETECTION (EXPERT ANNOTATIONS)\n');
fprintf('----------------------------------------------\n');

% Load expert annotations from MIT-BIH database
% These are cardiologist-verified R-peak locations
fprintf('Loading expert R-peak annotations...\n');
[ann_samples, ann_types] = rdann('208', 'atr');

fprintf('  Total annotations loaded: %d\n', length(ann_samples));

% Filter to only keep QRS annotations (actual beats)
% Common beat types in MIT-BIH:
%   N = Normal beat
%   L = Left bundle branch block beat
%   R = Right bundle branch block beat
%   V = Premature ventricular contraction
%   A = Atrial premature beat
%   / = Paced beat
%   f = Fusion of ventricular and normal beat
%   ! = Ventricular flutter wave
valid_beat_types = ['N', 'L', 'R', 'V', 'A', '/', 'f', '!', 'E', 'j'];
valid_beats_mask = ismember(ann_types, valid_beat_types);

R_peak_locations = ann_samples(valid_beats_mask);
R_peak_annotations = ann_types(valid_beats_mask);

fprintf('  Valid beat annotations: %d\n', length(R_peak_locations));
fprintf('  Annotation types present: %s\n', unique(R_peak_annotations)');

% Calculate R-peak times and values
R_peak_times = R_peak_locations / fs;
R_peak_values = preprocessed_signal(R_peak_locations);

num_beats = length(R_peak_locations);
fprintf('\n  Total R-peaks (beats): %d\n', num_beats);
fprintf('  Average heart rate: %.1f BPM\n\n', (num_beats / recording_duration_min));

% -------------------------------------------------------------------------
% Visualize R-Peak Detection
% -------------------------------------------------------------------------

% Figure: R-Peak Detection Overview (First 10 seconds)
figure('Name', 'R-Peak Detection with Annotations');

samples_10s = 10 * fs;
subplot(2,1,1);
plot(tm(1:samples_10s), preprocessed_signal(1:samples_10s), 'c', 'LineWidth', 0.8);
hold on;
peaks_in_window = R_peak_locations(R_peak_locations <= samples_10s);
plot(tm(peaks_in_window), preprocessed_signal(peaks_in_window), ...
    'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold off;
title('ECG with Expert-Annotated R-Peaks (First 10 seconds)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
legend('Preprocessed ECG', 'R-Peaks (Expert Annotations)');
grid on;

% Zoom to 3 seconds for better detail
subplot(2,1,2);
samples_3s = 3 * fs;
plot(tm(1:samples_3s), preprocessed_signal(1:samples_3s), 'c', 'LineWidth', 1);
hold on;
peaks_in_3s = R_peak_locations(R_peak_locations <= samples_3s);
plot(tm(peaks_in_3s), preprocessed_signal(peaks_in_3s), ...
    'rv', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold off;
title('R-Peak Detection - Zoomed View (First 3 seconds)');
xlabel('Time (Seconds)');
ylabel('Amplitude (mV)');
legend('Preprocessed ECG', 'R-Peaks');
grid on;

fprintf('R-peak detection complete!\n\n');

% -------------------------------------------------------------------------
% PART 3: BEAT SEGMENTATION
% -------------------------------------------------------------------------

fprintf('PART 3: BEAT SEGMENTATION\n');
fprintf('-------------------------\n');

% Define segmentation window around each R-peak
pre_R_sec = 0.2;    % 200 ms before R-peak
post_R_sec = 0.4;   % 400 ms after R-peak
pre_R_samples = round(pre_R_sec * fs);
post_R_samples = round(post_R_sec * fs);
total_window_samples = pre_R_samples + post_R_samples + 1;

fprintf('Segmentation window:\n');
fprintf('  Pre-R:  %.0f ms (%d samples)\n', pre_R_sec*1000, pre_R_samples);
fprintf('  Post-R: %.0f ms (%d samples)\n', post_R_sec*1000, post_R_samples);
fprintf('  Total:  %.0f ms (%d samples)\n\n', (pre_R_sec+post_R_sec)*1000, total_window_samples);

% Extract beats
beat_matrix = zeros(total_window_samples, num_beats);
valid_beat_indices = zeros(num_beats, 1);  % ADD THIS LINE - Preallocate
valid_beat_count = 0;

fprintf('Extracting individual beats...\n');
for i = 1:num_beats
    R_idx = R_peak_locations(i);
    start_idx = R_idx - pre_R_samples;
    end_idx = R_idx + post_R_samples;
    
    % Check if window is within signal bounds
    if start_idx >= 1 && end_idx <= length(preprocessed_signal)
        valid_beat_count = valid_beat_count + 1;
        beat_matrix(:, valid_beat_count) = preprocessed_signal(start_idx:end_idx);
        valid_beat_indices(valid_beat_count) = i;
    end
end

% Trim both arrays to valid beats only
beat_matrix = beat_matrix(:, 1:valid_beat_count);
valid_beat_indices = valid_beat_indices(1:valid_beat_count);  % ADD THIS LINE - Trim

fprintf('  Valid beats extracted: %d / %d\n', valid_beat_count, num_beats);
fprintf('  Excluded: %d (boundary issues)\n\n', num_beats - valid_beat_count);

% Create time axis for beats (in milliseconds, centered at R-peak)
beat_time_axis_ms = (-pre_R_samples:post_R_samples) * (1000/fs);

% Calculate average beat
average_beat = mean(beat_matrix, 2);
std_beat = std(beat_matrix, 0, 2);

% -------------------------------------------------------------------------
% Calculate Average NORMAL Beat and Correlations
% -------------------------------------------------------------------------

fprintf('Calculating beat correlations with average normal beat...\n');

% Extract only NORMAL beats (type 'N') to create template
normal_beat_mask = (R_peak_annotations(valid_beat_indices) == 'N');
normal_beats = beat_matrix(:, normal_beat_mask);
num_normal_beats = sum(normal_beat_mask);

fprintf('  Normal beats found: %d / %d (%.1f%%)\n', num_normal_beats, valid_beat_count, ...
    100*num_normal_beats/valid_beat_count);

% Calculate average NORMAL beat (the template)
if num_normal_beats > 0
    average_normal_beat = mean(normal_beats, 2);
    fprintf('  Average normal beat calculated from %d beats\n', num_normal_beats);
else
    % Fallback: use overall average if no normal beats
    average_normal_beat = average_beat;
    fprintf('  Warning: No normal beats found, using overall average\n');
end

% Compute correlation of each beat with average normal beat
beat_correlations = zeros(valid_beat_count, 1);

for i = 1:valid_beat_count
    % Compute Pearson correlation coefficient
    beat_correlations(i) = corr(beat_matrix(:, i), average_normal_beat);
end

% Correlation statistics
mean_corr = mean(beat_correlations);
std_corr = std(beat_correlations);
min_corr = min(beat_correlations);
max_corr = max(beat_correlations);

fprintf('  Correlation statistics:\n');
fprintf('    Mean:   %.4f\n', mean_corr);
fprintf('    Std:    %.4f\n', std_corr);
fprintf('    Min:    %.4f\n', min_corr);
fprintf('    Max:    %.4f\n', max_corr);

% Find beats with low correlation (potential abnormal beats)
low_corr_threshold = mean_corr - 2*std_corr;
low_corr_beats = find(beat_correlations < low_corr_threshold);
fprintf('    Beats with correlation < %.3f (mean-2*std): %d\n', ...
    low_corr_threshold, length(low_corr_beats));

% Correlation by beat type
unique_types = unique(R_peak_annotations(valid_beat_indices));
fprintf('\n  Correlation by beat type:\n');
for i = 1:length(unique_types)
    type_mask = (R_peak_annotations(valid_beat_indices) == unique_types(i));
    type_corr = beat_correlations(type_mask);
    fprintf('    %c: %.4f ± %.4f (n=%d)\n', unique_types(i), ...
        mean(type_corr), std(type_corr), length(type_corr));
end
fprintf('\n');

% -------------------------------------------------------------------------
% Visualize Beat Segmentation
% -------------------------------------------------------------------------

% Figure: Beat Overlay with Average
figure('Name', 'Beat Overlay with Average');

% Plot all beats in light gray
plot(beat_time_axis_ms, beat_matrix, 'Color', [0.7 0.7 0.7 0.1]);
hold on;

% Overlay average beat in color
plot(beat_time_axis_ms, average_beat, 'r', 'LineWidth', 2.5);

% Mark R-peak location
xline(0, 'c--', 'R-peak', 'LineWidth', 1.5);

hold off;
title(sprintf('Beat Overlay - All %d Beats with Average', valid_beat_count));
xlabel('Time Relative to R-Peak (ms)');
ylabel('Amplitude (mV)');
legend('Individual Beats', 'Average Beat', 'R-Peak Location');
grid on;

% Figure: Average Beat with Confidence Intervals
figure('Name', 'Average Beat Analysis');

% Plot confidence interval (mean ± std)
fill([beat_time_axis_ms, fliplr(beat_time_axis_ms)], ...
     [average_beat' + std_beat', fliplr(average_beat' - std_beat')], ...
     'c', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;

% Plot average beat
plot(beat_time_axis_ms, average_beat, 'b', 'LineWidth', 2);

% Mark characteristic points
xline(0, 'r--', 'R', 'LineWidth', 1.5);

hold off;
title('Average Beat with Standard Deviation');
xlabel('Time Relative to R-Peak (ms)');
ylabel('Amplitude (mV)');
legend('Mean ± Std', 'Average Beat', 'R-Peak');
grid on;

% Figure: Sample Individual Beats
figure('Name', 'Sample Individual Beats');

num_sample_beats = min(9, valid_beat_count);
for i = 1:num_sample_beats
    subplot(3, 3, i);
    plot(beat_time_axis_ms, beat_matrix(:, i), 'c', 'LineWidth', 1);
    hold on;
    xline(0, 'r--', 'R', 'LineWidth', 1);
    hold off;
    title(sprintf('Beat %d', i));
    xlabel('Time (ms)');
    ylabel('mV');
    grid on;
end
sgtitle('Sample Individual Beats');

% -------------------------------------------------------------------------
% Visualize Beat Correlations
% -------------------------------------------------------------------------

% Figure: Beat Correlation Analysis
figure('Name', 'Beat Correlation Analysis', 'Position', [100 100 1400 800]);

% Subplot 1: Correlation Distribution
subplot(2,3,1);
histogram(beat_correlations, 40, 'FaceColor', [0.3 0.6 0.8], 'EdgeColor', 'k');
hold on;
xline(mean_corr, 'r--', sprintf('Mean: %.3f', mean_corr), 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
xline(mean_corr - 2*std_corr, 'r:', '2σ', 'LineWidth', 1.5);
hold off;
title('Correlation Distribution');
xlabel('Correlation with Average Normal Beat');
ylabel('Count');
grid on;

% Subplot 2: Correlation by Beat Type (Box Plot)
subplot(2,3,2);
beat_types_for_corr = R_peak_annotations(valid_beat_indices);
boxplot(beat_correlations, beat_types_for_corr, 'Colors', 'c');
title('Correlation by Beat Type');
ylabel('Correlation');
xlabel('Beat Type');
grid on;

% Subplot 3: Correlation vs Beat Index (Time Series)
subplot(2,3,3);
scatter(1:valid_beat_count, beat_correlations, 10, beat_correlations, 'filled');
colormap(jet);
colorbar;
hold on;
yline(mean_corr, 'r--', 'Mean', 'LineWidth', 1.5);
yline(low_corr_threshold, 'r:', '2σ below', 'LineWidth', 1);
hold off;
title('Correlation Over Time');
xlabel('Beat Index');
ylabel('Correlation');
grid on;

% Subplot 4: Average Normal Beat vs Overall Average
subplot(2,3,4);
plot(beat_time_axis_ms, average_normal_beat, 'c', 'LineWidth', 2);
hold on;
plot(beat_time_axis_ms, average_beat, 'r--', 'LineWidth', 1.5);
xline(0, 'k--', 'R', 'LineWidth', 1);
hold off;
title('Average Normal Beat (Template)');
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
legend('Avg Normal (N)', 'Avg All Beats', 'R-peak', 'Location', 'best');
grid on;

% Subplot 5: High Correlation Examples
subplot(2,3,5);
[~, high_corr_idx] = maxk(beat_correlations, 5);
h_high = [];  % Store handles
for i = 1:min(5, length(high_corr_idx))
    h = plot(beat_time_axis_ms, beat_matrix(:, high_corr_idx(i)), ...
        'LineWidth', 1.5, 'Color', [0 0.6 0 0.6]);  % Darker green, less transparent
    if i == 1
        h_high = h;  % Save first handle for legend
    end
    hold on;
end
h_template = plot(beat_time_axis_ms, average_normal_beat, 'c', 'LineWidth', 2.5);  % Blue, thicker
xline(0, 'k--', 'R', 'LineWidth', 1);
hold off;
title(sprintf('High Correlation Beats (r > %.3f)', prctile(beat_correlations, 95)));
xlabel('Time (ms)');
ylabel('mV');
legend([h_high, h_template], {'High Corr Beats', 'Normal Template'}, 'Location', 'best');
grid on;

% Subplot 6: Low Correlation Examples
subplot(2,3,6);
[~, low_corr_idx] = mink(beat_correlations, 5);
h_low = [];  % Store handles
for i = 1:min(5, length(low_corr_idx))
    h = plot(beat_time_axis_ms, beat_matrix(:, low_corr_idx(i)), ...
        'LineWidth', 1.5, 'Color', [0.8 0 0 0.6]);  % Darker red, less transparent
    if i == 1
        h_low = h;  % Save first handle for legend
    end
    hold on;
end
h_template = plot(beat_time_axis_ms, average_normal_beat, 'c', 'LineWidth', 2.5);  % Blue, thicker
xline(0, 'k--', 'R', 'LineWidth', 1);
hold off;
title(sprintf('Low Correlation Beats (r < %.3f)', prctile(beat_correlations, 5)));
xlabel('Time (ms)');
ylabel('mV');
legend([h_low, h_template], {'Low Corr Beats', 'Normal Template'}, 'Location', 'best');
grid on;

% Remove overlapping titles - just use figure window name
% The 6 subplots are self-explanatory without a main title

% Figure: Correlation Scatter by Beat Type
figure('Name', 'Correlation by Beat Type');

unique_types = unique(R_peak_annotations(valid_beat_indices));
colors = lines(length(unique_types));

for i = 1:length(unique_types)
    type_mask = (R_peak_annotations(valid_beat_indices) == unique_types(i));
    type_indices = find(type_mask);
    type_corr = beat_correlations(type_mask);
    
    scatter(type_indices, type_corr, 30, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
end

yline(mean_corr, 'r--', sprintf('Overall Mean: %.3f', mean_corr), 'LineWidth', 2);
yline(low_corr_threshold, 'r:', 'Outlier Threshold', 'LineWidth', 1.5);
hold off;

title('Beat Correlation by Type and Time');
xlabel('Beat Index');
ylabel('Correlation with Normal Template');
legend([unique_types', {'Overall Mean', 'Threshold'}], 'Location', 'best');
grid on;
ylim([min(beat_correlations)-0.1, 1]);

fprintf('Beat segmentation complete!\n\n');

% -------------------------------------------------------------------------
% PART 4: HEART RATE ANALYSIS
% -------------------------------------------------------------------------

fprintf('PART 4: HEART RATE ANALYSIS\n');
fprintf('---------------------------\n');

% Calculate RR intervals (time between consecutive R-peaks)
RR_intervals_samples = diff(R_peak_locations);
RR_intervals_sec = RR_intervals_samples / fs;
RR_intervals_ms = RR_intervals_sec * 1000;

num_RR_original = length(RR_intervals_ms);
fprintf('Total RR intervals: %d\n', num_RR_original);

% Filter RR intervals to remove outliers
% Physiological limits: 400ms to 1500ms (40-150 BPM)
min_RR_ms = 400;    % 150 BPM
max_RR_ms = 1500;   % 40 BPM

valid_RR_mask = (RR_intervals_ms >= min_RR_ms) & (RR_intervals_ms <= max_RR_ms);
RR_intervals_ms = RR_intervals_ms(valid_RR_mask);

too_short = sum(RR_intervals_samples / fs * 1000 < min_RR_ms);
too_long = sum(RR_intervals_samples / fs * 1000 > max_RR_ms);
total_outliers = too_short + too_long;

fprintf('  Outliers removed: %d (%.1f%%)\n', total_outliers, 100*total_outliers/num_RR_original);
fprintf('    Too short (< %d ms): %d\n', min_RR_ms, too_short);
fprintf('    Too long (> %d ms): %d\n\n', max_RR_ms, too_long);

num_RR = length(RR_intervals_ms);

% Calculate instantaneous heart rate from filtered RR intervals
heart_rate_bpm = 60000 ./ RR_intervals_ms;

% Time axis for RR intervals
RR_time_axis = R_peak_times(1:num_RR);

% Calculate heart rate statistics
average_HR_correct = num_beats / recording_duration_min;
median_HR = median(heart_rate_bpm);
min_HR = min(heart_rate_bpm);
max_HR = max(heart_rate_bpm);
std_HR = std(heart_rate_bpm);

fprintf('Heart Rate Statistics (Overall):\n');
fprintf('  Average HR: %.1f BPM (%d beats / %.1f min)\n', average_HR_correct, num_beats, recording_duration_min);
fprintf('  Median HR:  %.1f BPM\n', median_HR);
fprintf('  Min HR:     %.1f BPM\n', min_HR);
fprintf('  Max HR:     %.1f BPM\n', max_HR);
fprintf('  Std HR:     %.1f BPM\n\n', std_HR);

% -------------------------------------------------------------------------
% Calculate Windowed Heart Rate (Sliding 30-Second Windows, 1-Second Step)
% -------------------------------------------------------------------------
% Method: 
%   1. Use sliding 30-second windows, shifted by 1 second each time
%   2. Calculate heart rate for each window (beats in window / 0.5 minutes)
%   3. Average all window heart rates for final result
%
% Example:
%   Window 1: seconds 0-30  -> calculate HR
%   Window 2: seconds 1-31  -> calculate HR
%   Window 3: seconds 2-32  -> calculate HR
%   ... continue until end of signal
%   Final HR = mean of all window HRs

fprintf('Calculating windowed heart rate (sliding window method)...\n');

% Parameters
window_duration_sec = 30;  % 30-second windows
window_step_sec = 1;       % 1-second step (shift)

% Total recording duration in seconds
total_duration_sec = floor(length(preprocessed_signal) / fs);

% Calculate number of windows
num_windows = total_duration_sec - window_duration_sec + 1;

fprintf('  Recording duration: %d seconds (%.1f minutes)\n', total_duration_sec, total_duration_sec/60);
fprintf('  Window duration: %d seconds\n', window_duration_sec);
fprintf('  Window step: %d second(s)\n', window_step_sec);
fprintf('  Total sliding windows: %d\n\n', num_windows);

% Convert R-peak locations to time in seconds
R_peak_times_sec = R_peak_locations / fs;

% Initialize arrays for sliding window results
window_HR = NaN(num_windows, 1);
window_times = zeros(num_windows, 1);
window_beat_counts = zeros(num_windows, 1);

fprintf('  Processing sliding windows...\n');

% Calculate heart rate for each sliding window
for w = 1:num_windows
    % Define window boundaries (in seconds)
    window_start_sec = (w - 1);  % 0-indexed start
    window_end_sec = window_start_sec + window_duration_sec;
    
    % Center time of window (for plotting)
    window_times(w) = (window_start_sec + window_end_sec) / 2;
    
    % Find R-peaks within this window
    peaks_in_window = (R_peak_times_sec >= window_start_sec) & (R_peak_times_sec < window_end_sec);
    num_beats_in_window = sum(peaks_in_window);
    window_beat_counts(w) = num_beats_in_window;
    
    % Calculate heart rate for this window
    % HR = (number of beats) / (window duration in minutes)
    if num_beats_in_window >= 2
        window_duration_min = window_duration_sec / 60;  % 0.5 minutes for 30-sec window
        window_HR(w) = num_beats_in_window / window_duration_min;
    end
end

% Identify valid windows (those with at least 2 beats)
valid_windows_mask = ~isnan(window_HR);
num_valid_windows = sum(valid_windows_mask);

fprintf('    Valid windows: %d / %d (%.1f%%)\n', ...
    num_valid_windows, num_windows, 100*num_valid_windows/num_windows);

% -------------------------------------------------------------------------
% Calculate Final Average Heart Rate
% -------------------------------------------------------------------------

% Average all valid window heart rates
final_avg_HR_windowed = mean(window_HR(valid_windows_mask));

% Create compatible variables for downstream code
window_hr_mean = window_HR(valid_windows_mask);
window_times_valid = window_times(valid_windows_mask);

% Calculate statistics
window_hr_std_overall = std(window_HR(valid_windows_mask));
window_hr_min = min(window_HR(valid_windows_mask));
window_hr_max = max(window_HR(valid_windows_mask));

fprintf('\nWindowed Heart Rate Results:\n');
fprintf('  Window HR range:    %.1f - %.1f BPM\n', window_hr_min, window_hr_max);
fprintf('  Window HR std:      %.1f BPM\n', window_hr_std_overall);
fprintf('  \n');
fprintf('  >>> FINAL AVERAGE HR (sliding window method): %.1f BPM <<<\n', final_avg_HR_windowed);
fprintf('      (averaged from %d sliding %d-sec windows, %d-sec step)\n\n', ...
    num_valid_windows, window_duration_sec, window_step_sec);
fprintf('  Comparison:\n');
fprintf('    Traditional (beats/duration): %.1f BPM\n', average_HR_correct);
fprintf('    Sliding window method:        %.1f BPM\n', final_avg_HR_windowed);
fprintf('    Difference:                   %.2f BPM\n\n', ...
    abs(final_avg_HR_windowed - average_HR_correct));

% -------------------------------------------------------------------------
% Visualize Heart Rate Analysis
% -------------------------------------------------------------------------

% Figure: RR Intervals and Heart Rate Trends
figure('Name', 'Heart Rate Analysis');

% Subplot 1: RR Intervals (Tachogram)
subplot(3,1,1);
plot(RR_time_axis/60, RR_intervals_ms, 'c', 'LineWidth', 0.5);
title('RR Interval Tachogram');
xlabel('Time (minutes)');
ylabel('RR Interval (ms)');
grid on;

% Subplot 2: Instantaneous Heart Rate
subplot(3,1,2);
plot(RR_time_axis/60, heart_rate_bpm, 'Color', [0.2 0.6 0.2], 'LineWidth', 0.5);
hold on;
yline(average_HR_correct, 'r--', sprintf('Traditional Avg: %.1f BPM', average_HR_correct), 'LineWidth', 1.5);
hold off;
title('Instantaneous Heart Rate');
xlabel('Time (minutes)');
ylabel('Heart Rate (BPM)');
grid on;

% Subplot 3: Sliding Window Heart Rate
subplot(3,1,3);
plot(window_times_valid/60, window_hr_mean, 'c-', 'LineWidth', 1);
hold on;
yline(final_avg_HR_windowed, 'r--', sprintf('Sliding Window Avg: %.1f BPM', final_avg_HR_windowed), 'LineWidth', 2);
hold off;
title(sprintf('Sliding Window Heart Rate (%d-sec windows, %d-sec step)', window_duration_sec, window_step_sec));
xlabel('Time (minutes)');
ylabel('Heart Rate (BPM)');
legend('Window HR', sprintf('Final Avg: %.1f BPM', final_avg_HR_windowed), 'Location', 'best');
grid on;

% Figure: Detailed Sliding Window Analysis
figure('Name', 'Sliding Window Heart Rate Analysis (Detailed)', 'Position', [100 100 1200 800]);

% Subplot 1: All sliding window heart rates
subplot(3,1,1);
plot(window_times/60, window_HR, 'b-', 'LineWidth', 0.5);
hold on;
yline(final_avg_HR_windowed, 'r--', 'LineWidth', 2);
xlabel('Time (minutes)');
ylabel('Heart Rate (BPM)');
title(sprintf('Sliding Window HR (%d-sec windows, %d-sec step) - All %d Windows', ...
    window_duration_sec, window_step_sec, num_windows));
legend('Window HR', sprintf('Final Avg: %.1f BPM', final_avg_HR_windowed), 'Location', 'best');
grid on;

% Subplot 2: Distribution of window heart rates
subplot(3,1,2);
histogram(window_HR(valid_windows_mask), 30, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k');
hold on;
xline(final_avg_HR_windowed, 'r--', sprintf('Mean: %.1f BPM', final_avg_HR_windowed), 'LineWidth', 2);
xline(median(window_HR(valid_windows_mask)), 'g--', sprintf('Median: %.1f BPM', median(window_HR(valid_windows_mask))), 'LineWidth', 1.5);
hold off;
xlabel('Heart Rate (BPM)');
ylabel('Count');
title('Distribution of Sliding Window Heart Rates');
grid on;

% Subplot 3: Comparison of methods
subplot(3,1,3);
methods = categorical({'Sliding Window Method', 'Traditional (beats/duration)'});
methods = reordercats(methods, {'Sliding Window Method', 'Traditional (beats/duration)'});
bar(methods, [final_avg_HR_windowed, average_HR_correct], 'FaceColor', [0.2 0.7 0.5]);
ylabel('Heart Rate (BPM)');
title('Comparison of Heart Rate Calculation Methods');
text(1, final_avg_HR_windowed + 1, sprintf('%.1f BPM', final_avg_HR_windowed), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, average_HR_correct + 1, sprintf('%.1f BPM', average_HR_correct), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

sgtitle(sprintf('Heart Rate Analysis: Sliding %d-Second Windows (1-Second Step)', window_duration_sec), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Figure: Distribution Histograms and Box Plots
figure('Name', 'Heart Rate Distributions');

% RR Interval Histogram
subplot(2,2,1);
histogram(RR_intervals_ms, 30, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'k');
title('RR Interval Distribution');
xlabel('RR Interval (ms)');
ylabel('Count');
grid on;

% Heart Rate Histogram
subplot(2,2,2);
histogram(heart_rate_bpm, 30, 'FaceColor', [0.2 0.7 0.3], 'EdgeColor', 'k');
title('Heart Rate Distribution');
xlabel('Heart Rate (BPM)');
ylabel('Count');
grid on;

% RR Interval Box Plot
subplot(2,2,3);
boxplot(RR_intervals_ms, 'Orientation', 'horizontal', 'Colors', 'c');
title('RR Interval Box Plot');
xlabel('RR Interval (ms)');
set(gca, 'YTickLabel', {''});
grid on;

% Heart Rate Box Plot
subplot(2,2,4);
boxplot(heart_rate_bpm, 'Orientation', 'horizontal', 'Colors', 'g');
title('Heart Rate Box Plot');
xlabel('Heart Rate (BPM)');
set(gca, 'YTickLabel', {''});
grid on;

% Figure: Poincaré Plot
figure('Name', 'Poincaré Plot');

RR_n = RR_intervals_ms(1:end-1);
RR_n1 = RR_intervals_ms(2:end);

scatter(RR_n, RR_n1, 5, 'c', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

min_RR_plot = min([RR_n; RR_n1]);
max_RR_plot = max([RR_n; RR_n1]);
plot([min_RR_plot max_RR_plot], [min_RR_plot max_RR_plot], 'r--', 'LineWidth', 1.5);

hold off;
title('Poincaré Plot (RR_{n} vs RR_{n+1})');
xlabel('RR_n (ms)');
ylabel('RR_{n+1} (ms)');
axis equal;
grid on;
legend('RR intervals', 'Line of identity', 'Location', 'southeast');

fprintf('Heart rate analysis complete!\n\n');

% -------------------------------------------------------------------------
% PART 5: HRV METRICS
% -------------------------------------------------------------------------

fprintf('PART 5: HRV (HEART RATE VARIABILITY) METRICS\n');
fprintf('---------------------------------------------\n');

% Time-domain HRV metrics
mean_RR_ms = mean(RR_intervals_ms);
SDNN_ms = std(RR_intervals_ms);

% RMSSD: Root mean square of successive differences
successive_diff = diff(RR_intervals_ms);
RMSSD_ms = sqrt(mean(successive_diff.^2));

% pNN50: Percentage of successive RR intervals that differ by more than 50 ms
NN50_count = sum(abs(successive_diff) > 50);
pNN50 = 100 * NN50_count / length(successive_diff);

fprintf('Time-Domain HRV Metrics:\n');
fprintf('  Mean RR:  %.1f ms\n', mean_RR_ms);
fprintf('  SDNN:     %.1f ms (standard deviation of all RR intervals)\n', SDNN_ms);
fprintf('  RMSSD:    %.1f ms (root mean square of successive differences)\n', RMSSD_ms);
fprintf('  NN50:     %d intervals\n', NN50_count);
fprintf('  pNN50:    %.1f%% (percentage of successive RR diffs > 50ms)\n\n', pNN50);

% -------------------------------------------------------------------------
% Frequency-Domain HRV Analysis
% -------------------------------------------------------------------------
% The RR interval series is unevenly sampled (intervals vary), so we need
% to interpolate to create an evenly-sampled signal for spectral analysis.
%
% Standard frequency bands (Task Force of ESC/NASPE, 1996):
%   VLF: 0.003 - 0.04 Hz  (Very Low Frequency) - Thermoregulation, hormonal
%   LF:  0.04  - 0.15 Hz  (Low Frequency)  - Sympathetic + Parasympathetic
%   HF:  0.15  - 0.40 Hz  (High Frequency) - Parasympathetic (respiratory)
%
% LF/HF Ratio: Indicator of sympathovagal balance

fprintf('Frequency-Domain HRV Analysis:\n');
fprintf('------------------------------\n');

% Create time axis for RR intervals (cumulative time)
RR_intervals_sec = RR_intervals_ms / 1000;
RR_cumtime = cumsum(RR_intervals_sec);  % Cumulative time in seconds
RR_cumtime = [0; RR_cumtime(1:end-1)];  % Start at 0

% Interpolation parameters
interp_fs = 4;  % 4 Hz is standard for HRV spectral analysis
interp_dt = 1 / interp_fs;

% Create evenly-spaced time vector
t_interp = (0:interp_dt:RR_cumtime(end))';

% Interpolate RR intervals to even sampling using cubic spline
RR_interp = interp1(RR_cumtime, RR_intervals_ms, t_interp, 'spline');

% Remove mean (detrend) for spectral analysis
RR_interp_detrend = RR_interp - mean(RR_interp);

fprintf('  Interpolation:\n');
fprintf('    Original RR samples: %d (unevenly spaced)\n', length(RR_intervals_ms));
fprintf('    Interpolated samples: %d at %.1f Hz\n', length(RR_interp), interp_fs);
fprintf('    Duration: %.1f seconds\n\n', t_interp(end));

% Compute Power Spectral Density using Welch's method
% Window length: ~256 seconds for good VLF resolution (or length/5)
window_length = min(256 * interp_fs, floor(length(RR_interp) / 5));
window_length = max(window_length, 64);  % Minimum window size
noverlap = floor(window_length / 2);     % 50% overlap
nfft = max(1024, 2^nextpow2(window_length));

[psd, f_psd] = pwelch(RR_interp_detrend, hamming(window_length), noverlap, nfft, interp_fs);

% Define frequency bands (Hz)
VLF_band = [0.003, 0.04];
LF_band = [0.04, 0.15];
HF_band = [0.15, 0.40];

% Calculate power in each band (integrate PSD)
% Find indices for each band
VLF_idx = (f_psd >= VLF_band(1)) & (f_psd < VLF_band(2));
LF_idx = (f_psd >= LF_band(1)) & (f_psd < LF_band(2));
HF_idx = (f_psd >= HF_band(1)) & (f_psd <= HF_band(2));

% Calculate power using trapezoidal integration (ms^2)
VLF_power = trapz(f_psd(VLF_idx), psd(VLF_idx));
LF_power = trapz(f_psd(LF_idx), psd(LF_idx));
HF_power = trapz(f_psd(HF_idx), psd(HF_idx));

% Total power (VLF + LF + HF)
total_power = VLF_power + LF_power + HF_power;

% Normalized units (exclude VLF for normalization, per standard)
LF_norm = 100 * LF_power / (LF_power + HF_power);  % n.u. (normalized units)
HF_norm = 100 * HF_power / (LF_power + HF_power);  % n.u.

% LF/HF Ratio
LF_HF_ratio = LF_power / HF_power;

fprintf('  Frequency Band Powers (ms^2):\n');
fprintf('    VLF (0.003-0.04 Hz): %.2f ms^2\n', VLF_power);
fprintf('    LF  (0.04-0.15 Hz):  %.2f ms^2\n', LF_power);
fprintf('    HF  (0.15-0.40 Hz):  %.2f ms^2\n', HF_power);
fprintf('    Total Power:         %.2f ms^2\n\n', total_power);

fprintf('  Normalized Units:\n');
fprintf('    LF (n.u.): %.1f%%\n', LF_norm);
fprintf('    HF (n.u.): %.1f%%\n\n', HF_norm);

fprintf('  Sympathovagal Balance:\n');
fprintf('    LF/HF Ratio: %.2f\n', LF_HF_ratio);
if LF_HF_ratio > 2
    fprintf('    Interpretation: Sympathetic dominance\n\n');
elseif LF_HF_ratio < 0.5
    fprintf('    Interpretation: Parasympathetic dominance\n\n');
else
    fprintf('    Interpretation: Balanced autonomic tone\n\n');
end

% -------------------------------------------------------------------------
% Visualize Frequency-Domain HRV
% -------------------------------------------------------------------------

% Figure: Frequency-Domain HRV Analysis
figure('Name', 'Frequency-Domain HRV Analysis', 'Position', [100 100 1400 900]);

% Subplot 1: Interpolated RR Interval Series
subplot(2,3,1);
plot(t_interp/60, RR_interp, 'c', 'LineWidth', 0.5);
xlabel('Time (minutes)');
ylabel('RR Interval (ms)');
title('Interpolated RR Interval Series');
grid on;

% Subplot 2: Power Spectral Density (Full)
subplot(2,3,2);
plot(f_psd, 10*log10(psd), 'c', 'LineWidth', 1);
xlabel('Frequency (Hz)');
ylabel('PSD (dB/Hz)');
title('Power Spectral Density');
xlim([0 0.5]);
grid on;

% Add vertical lines for band boundaries
hold on;
xline(VLF_band(2), 'r--', 'VLF|LF', 'LineWidth', 1);
xline(LF_band(2), 'g--', 'LF|HF', 'LineWidth', 1);
xline(HF_band(2), 'b--', 'HF', 'LineWidth', 1);
hold off;

% Subplot 3: PSD with Filled Bands (Linear Scale)
subplot(2,3,3);
hold on;

% Fill VLF band
fill([f_psd(VLF_idx); flipud(f_psd(VLF_idx))], ...
     [psd(VLF_idx); zeros(sum(VLF_idx), 1)], ...
     [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');

% Fill LF band
fill([f_psd(LF_idx); flipud(f_psd(LF_idx))], ...
     [psd(LF_idx); zeros(sum(LF_idx), 1)], ...
     [0.2 0.8 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');

% Fill HF band
fill([f_psd(HF_idx); flipud(f_psd(HF_idx))], ...
     [psd(HF_idx); zeros(sum(HF_idx), 1)], ...
     [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');

plot(f_psd, psd, 'k', 'LineWidth', 1);
hold off;

xlabel('Frequency (Hz)');
ylabel('PSD (ms^2/Hz)');
title('PSD with Frequency Bands');
xlim([0 0.5]);
legend('VLF', 'LF', 'HF', 'PSD', 'Location', 'northeast');
grid on;

% Subplot 4: Band Power Bar Chart
subplot(2,3,4);
band_powers = [VLF_power, LF_power, HF_power];
band_categories = categorical({'VLF', 'LF', 'HF'});
band_categories = reordercats(band_categories, {'VLF', 'LF', 'HF'});  % Prevent alphabetical sorting
b = bar(band_categories, band_powers, 'FaceColor', 'flat');
b.CData(1,:) = [0.8 0.2 0.2];  % VLF - red
b.CData(2,:) = [0.2 0.8 0.2];  % LF - green
b.CData(3,:) = [0.2 0.2 0.8];  % HF - blue
ylabel('Power (ms^2)');
title('Absolute Band Powers');
grid on;

% Add power values on bars
text(1, VLF_power + max(band_powers)*0.05, sprintf('%.1f', VLF_power), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, LF_power + max(band_powers)*0.05, sprintf('%.1f', LF_power), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(3, HF_power + max(band_powers)*0.05, sprintf('%.1f', HF_power), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Subplot 5: Normalized Power Pie Chart
subplot(2,3,5);
pie([LF_norm, HF_norm], {sprintf('LF: %.1f%%', LF_norm), sprintf('HF: %.1f%%', HF_norm)});
colormap([0.2 0.8 0.2; 0.2 0.2 0.8]);
title(sprintf('Normalized Powers (LF/HF = %.2f)', LF_HF_ratio));

% Subplot 6: HRV Summary Text
subplot(2,3,6);
axis off;

hrv_summary_text = {
    'FREQUENCY-DOMAIN HRV SUMMARY', ...
    '════════════════════════════════', ...
    '', ...
    'Absolute Powers (ms^2):', ...
    sprintf('  VLF (0.003-0.04 Hz): %.2f', VLF_power), ...
    sprintf('  LF  (0.04-0.15 Hz):  %.2f', LF_power), ...
    sprintf('  HF  (0.15-0.40 Hz):  %.2f', HF_power), ...
    sprintf('  Total Power:         %.2f', total_power), ...
    '', ...
    'Normalized Units:', ...
    sprintf('  LF (n.u.): %.1f%%', LF_norm), ...
    sprintf('  HF (n.u.): %.1f%%', HF_norm), ...
    '', ...
    'Sympathovagal Balance:', ...
    sprintf('  LF/HF Ratio: %.2f', LF_HF_ratio), ...
    '', ...
    'Clinical Interpretation:', ...
    '  VLF: Thermoregulation, hormones', ...
    '  LF:  Sympathetic + Parasympathetic', ...
    '  HF:  Parasympathetic (respiratory)', ...
    '  LF/HF: Autonomic balance indicator'
};

text(0.05, 0.95, hrv_summary_text, 'FontSize', 9, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Frequency-Domain Heart Rate Variability Analysis', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Frequency-domain HRV analysis complete!\n\n');

% -------------------------------------------------------------------------
% PART 6: MACHINE LEARNING CLASSIFICATION
% -------------------------------------------------------------------------

fprintf('PART 6: MACHINE LEARNING BEAT CLASSIFICATION\n');
fprintf('---------------------------------------------\n');

% Prepare data for ML classification
% Use the valid beats and their corresponding annotations
valid_annotations = R_peak_annotations(valid_beat_indices);

fprintf('Preparing data for machine learning...\n');
fprintf('  Total beats: %d\n', valid_beat_count);

% Count beat types
unique_types = unique(valid_annotations);
fprintf('  Beat types present:\n');
for i = 1:length(unique_types)
    count = sum(valid_annotations == unique_types(i));
    fprintf('    %c: %d beats\n', unique_types(i), count);
end

% Filter to only keep classes with sufficient samples (>= 10)
min_samples_per_class = 10;
class_counts = zeros(length(unique_types), 1);
for i = 1:length(unique_types)
    class_counts(i) = sum(valid_annotations == unique_types(i));
end

sufficient_classes = unique_types(class_counts >= min_samples_per_class);
valid_class_mask = ismember(valid_annotations, sufficient_classes);

% Filter data
X = beat_matrix(:, valid_class_mask)';  % Features: each row is a beat
y = valid_annotations(valid_class_mask); % Labels

fprintf('\n  After filtering (>= %d samples per class):\n', min_samples_per_class);
fprintf('  Classes kept: %s\n', sufficient_classes');
fprintf('  Total samples: %d\n\n', size(X, 1));

% Split data: 70% training, 30% testing
rng(42); % For reproducibility
cv = cvpartition(y, 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

fprintf('Data split:\n');
fprintf('  Training set: %d samples\n', size(X_train, 1));
fprintf('  Testing set:  %d samples\n\n', size(X_test, 1));

% -------------------------------------------------------------------------
% Method 1: Decision Tree Classifier
% -------------------------------------------------------------------------

fprintf('Training Decision Tree Classifier...\n');
tic;
tree_model = fitctree(X_train, y_train, 'MaxNumSplits', 100);
dt_train_time = toc;

% Predictions
tic;
y_pred_tree = predict(tree_model, X_test);
dt_test_time = toc;

% Evaluation
dt_accuracy = sum(y_pred_tree == y_test) / length(y_test) * 100;
dt_cm = confusionmat(y_test, y_pred_tree, 'Order', sufficient_classes);

fprintf('  Training time: %.4f seconds\n', dt_train_time);
fprintf('  Testing time:  %.4f seconds\n', dt_test_time);
fprintf('  Accuracy: %.2f%%\n\n', dt_accuracy);

% -------------------------------------------------------------------------
% DECISION TREE ENHANCED VISUALIZATIONS
% -------------------------------------------------------------------------

fprintf('\n┌─────────────────────────────────────────────────────────┐\n');
fprintf('│  CREATING ENHANCED DECISION TREE VISUALIZATIONS         │\n');
fprintf('└─────────────────────────────────────────────────────────┘\n\n');

% -------------------------------------------------------------------------
% VISUALIZATION 1: DECISION TREE STRUCTURE
% -------------------------------------------------------------------------

fprintf('1. Creating tree structure visualization...\n');

% Get tree properties
num_nodes = tree_model.NumNodes;
num_leaves = sum(tree_model.Children(:,1) == 0);  % Nodes with no children are leaves

figure('Name', 'Decision Tree - Structure Visualization', 'Position', [100 100 1400 800]);
view(tree_model, 'Mode', 'graph');
title(sprintf('Decision Tree Structure (Nodes: %d, Leaves: %d, Accuracy: %.2f%%)', ...
    num_nodes, num_leaves, dt_accuracy), 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 2: FEATURE IMPORTANCE
% -------------------------------------------------------------------------

fprintf('2. Calculating and visualizing feature importance...\n');

% Calculate feature importance
imp = predictorImportance(tree_model);

% Create time axis for features
pre_R_samples = round(0.2 * 360);  % Assuming 360 Hz sampling
post_R_samples = round(0.4 * 360);
feature_time_ms = (-pre_R_samples:post_R_samples) / 360 * 1000;

figure('Name', 'Decision Tree - Feature Importance', 'Position', [100 100 1200 800]);

% Panel 1: Feature importance over time
subplot(2,2,1);
plot(feature_time_ms, imp, 'b-', 'LineWidth', 2);
hold on;
xline(0, 'r--', 'R-peak', 'LineWidth', 2);
title('Feature Importance Across Beat Window', 'FontWeight', 'bold');
xlabel('Time relative to R-peak (ms)');
ylabel('Importance Score');
grid on;

% Panel 2: Top 20 most important features
[sorted_imp, sorted_idx] = sort(imp, 'descend');
top_n = min(20, length(sorted_imp));

subplot(2,2,2);
barh(1:top_n, sorted_imp(1:top_n), 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'YDir', 'reverse');
ylabel('Feature Rank');
xlabel('Importance Score');
title(sprintf('Top %d Most Important Features', top_n), 'FontWeight', 'bold');
grid on;

% Panel 3: Feature importance by temporal region
subplot(2,2,3);
n_regions = 10;
region_size = floor(length(imp) / n_regions);
region_importance = zeros(n_regions, 1);
region_labels = cell(n_regions, 1);

for i = 1:n_regions
    start_idx = (i-1)*region_size + 1;
    end_idx = min(i*region_size, length(imp));
    region_importance(i) = mean(imp(start_idx:end_idx));
    
    start_time = feature_time_ms(start_idx);
    end_time = feature_time_ms(end_idx);
    region_labels{i} = sprintf('%.0f-%.0f', start_time, end_time);
end

barh(region_importance, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'YTickLabel', region_labels, 'YDir', 'reverse');
xlabel('Average Importance');
title('Feature Importance by Temporal Region (ms)', 'FontWeight', 'bold');
grid on;

% Panel 4: Cumulative importance
subplot(2,2,4);
cumulative_imp = cumsum(sorted_imp) / sum(sorted_imp) * 100;
plot(1:length(cumulative_imp), cumulative_imp, 'g-', 'LineWidth', 2);
hold on;
yline(90, 'r--', '90%', 'LineWidth', 1.5);
yline(95, 'b--', '95%', 'LineWidth', 1.5);
xlabel('Number of Features');
ylabel('Cumulative Importance (%)');
title('Cumulative Feature Importance', 'FontWeight', 'bold');
grid on;

n_90 = find(cumulative_imp >= 90, 1);
n_95 = find(cumulative_imp >= 95, 1);
legend('Cumulative', ...
    sprintf('%d features for 90%%', n_90), ...
    sprintf('%d features for 95%%', n_95), ...
    'Location', 'southeast');

sgtitle('Decision Tree Feature Importance Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 3: ENHANCED CONFUSION MATRIX
% -------------------------------------------------------------------------

fprintf('3. Creating enhanced confusion matrix visualizations...\n');

% Normalized confusion matrix
C_dt_norm = dt_cm ./ sum(dt_cm, 2) * 100;

% Get actual class labels from confusion matrix
% The confusion matrix may not include all sufficient_classes if some weren't in test set
unique_test_classes = unique([y_test; y_pred_tree]);
class_labels_cm = cell(length(unique_test_classes), 1);
for i = 1:length(unique_test_classes)
    class_labels_cm{i} = char(unique_test_classes(i));
end

figure('Name', 'Decision Tree - Enhanced Confusion Matrix', 'Position', [100 100 1400 600]);

% Panel 1: Raw counts
subplot(1,3,1);
h1 = heatmap(class_labels_cm, class_labels_cm, dt_cm);
h1.Title = 'Confusion Matrix (Counts)';
h1.XLabel = 'Predicted Class';
h1.YLabel = 'True Class';
h1.Colormap = parula;
h1.FontSize = 10;

% Panel 2: Normalized percentages
subplot(1,3,2);
h2 = heatmap(class_labels_cm, class_labels_cm, C_dt_norm);
h2.Title = 'Confusion Matrix (Normalized %)';
h2.XLabel = 'Predicted Class';
h2.YLabel = 'True Class';
h2.Colormap = hot;
h2.FontSize = 10;

% Panel 3: Per-class metrics
subplot(1,3,3);
axis off;

TP = diag(dt_cm);
FP = sum(dt_cm, 1)' - TP;
FN = sum(dt_cm, 2) - TP;

precision = TP ./ (TP + FP) * 100;
recall = TP ./ (TP + FN) * 100;
f1_score = 2 * (precision .* recall) ./ (precision + recall);

% Handle NaN
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1_score(isnan(f1_score)) = 0;

metrics_text = {'PER-CLASS METRICS', '═══════════════════', ''};
for i = 1:length(class_labels_cm)
    metrics_text{end+1} = sprintf('Class %s:', class_labels_cm{i});
    metrics_text{end+1} = sprintf('  Precision: %.1f%%', precision(i));
    metrics_text{end+1} = sprintf('  Recall:    %.1f%%', recall(i));
    metrics_text{end+1} = sprintf('  F1-Score:  %.1f%%', f1_score(i));
    metrics_text{end+1} = '';
end
metrics_text{end+1} = sprintf('Overall Accuracy: %.2f%%', dt_accuracy);

text(0.1, 0.95, metrics_text, 'FontSize', 10, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree Classification Performance', 'FontSize', 14, 'FontWeight', 'bold');

% ==-------------------------------------------------------------------------
% VISUALIZATION 4: ROC CURVES (Multi-class One-vs-Rest)
% -------------------------------------------------------------------------

fprintf('4. Generating ROC curves...\n');

% Get decision scores
[~, scores_dt] = predict(tree_model, X_test);

% Use actual classes present in test set
unique_test_classes = unique([y_test; y_pred_tree]);
n_classes = length(unique_test_classes);

figure('Name', 'Decision Tree - ROC Curves', 'Position', [100 100 1200 800]);

colors = lines(n_classes);
auc_values = zeros(n_classes, 1);

subplot(2,2,[1 2]);
hold on;

for i = 1:n_classes
    binary_labels = (y_test == unique_test_classes(i));
    class_scores = scores_dt(:, i);
    
    [X_roc, Y_roc, ~, AUC] = perfcurve(binary_labels, class_scores, true);
    auc_values(i) = AUC;
    
    plot(X_roc, Y_roc, 'LineWidth', 2, 'Color', colors(i,:), ...
        'DisplayName', sprintf('Class %s (AUC=%.3f)', char(unique_test_classes(i)), AUC));
end

plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'DisplayName', 'Random');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves (One-vs-Rest)', 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 9);
grid on;
axis square;

% Panel 2: AUC comparison
subplot(2,2,3);
barh(auc_values, 'FaceColor', [0.3 0.6 0.9]);
class_labels_cell = cell(n_classes, 1);
for i = 1:n_classes
    class_labels_cell{i} = char(unique_test_classes(i));
end
set(gca, 'YTickLabel', class_labels_cell, 'YDir', 'reverse');
xlabel('AUC');
title('AUC by Class', 'FontWeight', 'bold');
xlim([0 1]);
grid on;

% Panel 3: Statistics
subplot(2,2,4);
axis off;

mean_auc = mean(auc_values);
std_auc = std(auc_values);

roc_text = {
    'ROC STATISTICS', ...
    '══════════════', ...
    '', ...
    sprintf('Mean AUC: %.4f', mean_auc), ...
    sprintf('Std AUC:  %.4f', std_auc), ...
    sprintf('Min AUC:  %.4f', min(auc_values)), ...
    sprintf('Max AUC:  %.4f', max(auc_values)), ...
    '', ...
    'INTERPRETATION:', ...
    '──────────────', ...
    'AUC = 1.0: Perfect', ...
    'AUC > 0.9: Excellent', ...
    'AUC > 0.8: Good', ...
    'AUC > 0.7: Fair', ...
    'AUC = 0.5: Random'
};

text(0.1, 0.95, roc_text, 'FontSize', 11, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree - ROC Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 5: PRECISION-RECALL CURVES
% -------------------------------------------------------------------------

fprintf('5. Generating Precision-Recall curves...\n');

figure('Name', 'Decision Tree - Precision-Recall Curves', 'Position', [100 100 1200 800]);

ap_values = zeros(n_classes, 1);

subplot(2,2,[1 2]);
hold on;

for i = 1:n_classes
    binary_labels = (y_test == unique_test_classes(i));
    class_scores = scores_dt(:, i);
    
    [X_pr, Y_pr, ~, AUC_PR] = perfcurve(binary_labels, class_scores, true, ...
        'XCrit', 'reca', 'YCrit', 'prec');
    ap_values(i) = AUC_PR;
    
    plot(X_pr, Y_pr, 'LineWidth', 2, 'Color', colors(i,:), ...
        'DisplayName', sprintf('Class %s (AP=%.3f)', char(unique_test_classes(i)), AUC_PR));
end

xlabel('Recall');
ylabel('Precision');
title('Precision-Recall Curves', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
axis square;

% Panel 2: Average Precision comparison
subplot(2,2,3);
barh(ap_values, 'FaceColor', [0.9 0.6 0.3]);
set(gca, 'YTickLabel', class_labels_cell, 'YDir', 'reverse');
xlabel('Average Precision');
title('Average Precision by Class', 'FontWeight', 'bold');
xlim([0 1]);
grid on;

% Panel 3: Statistics
subplot(2,2,4);
axis off;

mean_ap = mean(ap_values);

pr_text = {
    'PR STATISTICS', ...
    '═════════════', ...
    '', ...
    sprintf('Mean AP: %.4f', mean_ap), ...
    sprintf('Std AP:  %.4f', std(ap_values)), ...
    sprintf('Min AP:  %.4f', min(ap_values)), ...
    sprintf('Max AP:  %.4f', max(ap_values)), ...
    '', ...
    'mAP (Mean Average', ...
    'Precision) is useful', ...
    'for imbalanced datasets'
};

text(0.1, 0.95, pr_text, 'FontSize', 11, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree - Precision-Recall Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 6: LEARNING CURVES
% -------------------------------------------------------------------------

fprintf('6. Generating learning curves...\n');

train_sizes = round(linspace(0.1, 1.0, 10) * size(X_train, 1));
train_scores = zeros(length(train_sizes), 1);
val_scores = zeros(length(train_sizes), 1);

for i = 1:length(train_sizes)
    n_train = train_sizes(i);
    
    idx = randperm(size(X_train, 1), n_train);
    X_subset = X_train(idx, :);
    y_subset = y_train(idx);
    
    temp_model = fitctree(X_subset, y_subset, 'MaxNumSplits', 100);
    
    y_pred_train = predict(temp_model, X_subset);
    train_scores(i) = sum(y_pred_train == y_subset) / length(y_subset) * 100;
    
    y_pred_val = predict(temp_model, X_test);
    val_scores(i) = sum(y_pred_val == y_test) / length(y_test) * 100;
end

figure('Name', 'Decision Tree - Learning Curves', 'Position', [100 100 1200 500]);

subplot(1,2,1);
plot(train_sizes, train_scores, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(train_sizes, val_scores, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Training Set Size');
ylabel('Accuracy (%)');
title('Learning Curve', 'FontWeight', 'bold');
legend('Training', 'Validation', 'Location', 'best');
grid on;

% Panel 2: Analysis
subplot(1,2,2);
axis off;

overfit_gap = train_scores - val_scores;
final_gap = overfit_gap(end);

learning_text = {
    'LEARNING ANALYSIS', ...
    '════════════════', ...
    '', ...
    sprintf('Final Train Acc:  %.2f%%', train_scores(end)), ...
    sprintf('Final Val Acc:    %.2f%%', val_scores(end)), ...
    sprintf('Overfitting Gap:  %.2f%%', final_gap), ...
    '', ...
    'INTERPRETATION:', ...
    '──────────────', ...
    'Gap < 5%:  Well-generalized', ...
    'Gap 5-10%: Slight overfitting', ...
    'Gap > 10%: Overfitting', ...
    '', ...
    'If validation plateaus:', ...
    '• More training data', ...
    '• Feature engineering', ...
    '• Different model'
};

text(0.1, 0.95, learning_text, 'FontSize', 10, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree Learning Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 7: TREE DEPTH ANALYSIS
% -------------------------------------------------------------------------

fprintf('7. Analyzing effect of tree depth...\n');

max_depths = [3, 5, 10, 20, 30, 50, 100];
depth_train_acc = zeros(length(max_depths), 1);
depth_test_acc = zeros(length(max_depths), 1);
depth_n_leaves = zeros(length(max_depths), 1);

for i = 1:length(max_depths)
    temp_model = fitctree(X_train, y_train, 'MaxNumSplits', max_depths(i));
    
    y_pred_train = predict(temp_model, X_train);
    depth_train_acc(i) = sum(y_pred_train == y_train) / length(y_train) * 100;
    
    y_pred_test = predict(temp_model, X_test);
    depth_test_acc(i) = sum(y_pred_test == y_test) / length(y_test) * 100;
    
    % Count leaves: nodes with no children
    depth_n_leaves(i) = sum(temp_model.Children(:,1) == 0);
end

figure('Name', 'Decision Tree - Depth Analysis', 'Position', [100 100 1200 800]);

% Accuracy vs depth
subplot(2,2,1);
plot(max_depths, depth_train_acc, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(max_depths, depth_test_acc, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Maximum Tree Depth');
ylabel('Accuracy (%)');
title('Accuracy vs Tree Depth', 'FontWeight', 'bold');
legend('Training', 'Testing', 'Location', 'best');
grid on;
set(gca, 'XScale', 'log');

% Number of leaves
subplot(2,2,2);
plot(max_depths, depth_n_leaves, 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Maximum Tree Depth');
ylabel('Number of Leaf Nodes');
title('Model Complexity', 'FontWeight', 'bold');
grid on;
set(gca, 'XScale', 'log');

% Overfitting gap
subplot(2,2,3);
gap = depth_train_acc - depth_test_acc;
bar(max_depths, gap, 'FaceColor', [0.8 0.4 0.4]);
xlabel('Maximum Tree Depth');
ylabel('Train-Test Gap (%)');
title('Overfitting Analysis', 'FontWeight', 'bold');
grid on;
set(gca, 'XScale', 'log');

% Optimal depth analysis
subplot(2,2,4);
axis off;

[best_test_acc, best_idx] = max(depth_test_acc);
optimal_depth = max_depths(best_idx);

depth_text = {
    'DEPTH ANALYSIS', ...
    '══════════════', ...
    '', ...
    sprintf('Optimal Depth:      %d', optimal_depth), ...
    sprintf('Best Test Acc:      %.2f%%', best_test_acc), ...
    sprintf('Overfitting Gap:    %.2f%%', gap(best_idx)), ...
    sprintf('Number of Leaves:   %d', depth_n_leaves(best_idx)), ...
    '', ...
    'RECOMMENDATIONS:', ...
    '────────────────', ...
    'Shallow (3-10):', ...
    '• Interpretable', ...
    '• Less overfitting', ...
    '', ...
    'Deep (>50):', ...
    '• Better fit', ...
    '• Risk overfitting'
};

text(0.1, 0.95, depth_text, 'FontSize', 10, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree Depth Optimization', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 8: PRUNING ANALYSIS
% -------------------------------------------------------------------------

fprintf('8. Analyzing tree pruning...\n');

% Create a sequence of pruned trees
[E, SE, N, best_level] = cvloss(tree_model, 'SubTrees', 'all');

figure('Name', 'Decision Tree - Pruning Analysis', 'Position', [100 100 1200 800]);

% CV error vs tree size
subplot(2,2,1);
plot(N, E, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(N, E + SE, 'r--', 'LineWidth', 1);
plot(N, E - SE, 'r--', 'LineWidth', 1);
[min_err, min_idx] = min(E);
plot(N(min_idx), min_err, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Tree Size (Leaves)');
ylabel('CV Error');
title('Pruning: CV Error vs Size', 'FontWeight', 'bold');
legend('CV Error', 'SE', 'Minimum', 'Location', 'best');
grid on;

% Compare pruned vs unpruned
dt_pruned = prune(tree_model, 'Level', best_level);

y_pred_unpruned = predict(tree_model, X_test);
acc_unpruned = sum(y_pred_unpruned == y_test) / length(y_test) * 100;

y_pred_pruned = predict(dt_pruned, X_test);
acc_pruned = sum(y_pred_pruned == y_test) / length(y_test) * 100;

subplot(2,2,2);
comparison = [acc_unpruned, acc_pruned];
bar(comparison, 'FaceColor', [0.4 0.7 0.9]);
set(gca, 'XTickLabel', {'Unpruned', 'Pruned'});
ylabel('Test Accuracy (%)');
title('Unpruned vs Pruned', 'FontWeight', 'bold');
ylim([min(comparison)*0.95, max(comparison)*1.05]);
grid on;

for i = 1:2
    text(i, comparison(i), sprintf('%.2f%%', comparison(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Tree size comparison
subplot(2,2,3);
% Count leaves for unpruned and pruned trees
size_unpruned = sum(tree_model.Children(:,1) == 0);
size_pruned = sum(dt_pruned.Children(:,1) == 0);
sizes = [size_unpruned, size_pruned];

bar(sizes, 'FaceColor', [0.9 0.6 0.3]);
set(gca, 'XTickLabel', {'Unpruned', 'Pruned'});
ylabel('Number of Leaves');
title('Model Complexity', 'FontWeight', 'bold');
grid on;

for i = 1:2
    text(i, sizes(i), sprintf('%d', sizes(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Summary
subplot(2,2,4);
axis off;

reduction = (sizes(1) - sizes(2)) / sizes(1) * 100;

prune_text = {
    'PRUNING SUMMARY', ...
    '═══════════════', ...
    '', ...
    'UNPRUNED:', ...
    sprintf('  Leaves: %d', sizes(1)), ...
    sprintf('  Acc: %.2f%%', acc_unpruned), ...
    '', ...
    'PRUNED:', ...
    sprintf('  Leaves: %d', sizes(2)), ...
    sprintf('  Acc: %.2f%%', acc_pruned), ...
    '', ...
    sprintf('Size Reduction: %.1f%%', reduction), ...
    sprintf('Acc Change: %+.2f%%', acc_pruned - acc_unpruned)
};

text(0.1, 0.95, prune_text, 'FontSize', 10, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree Pruning Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 9: DECISION BOUNDARIES (2D PCA)
% -------------------------------------------------------------------------

fprintf('9. Creating decision boundary visualization...\n');

[coeff, score, ~, ~, explained] = pca(X_train);
X_train_2d = score(:, 1:2);
X_test_2d = (X_test - mean(X_train)) * coeff(:, 1:2);

dt_2d = fitctree(X_train_2d, y_train, 'MaxNumSplits', 20);

figure('Name', 'Decision Tree - Decision Boundaries', 'Position', [100 100 1200 800]);

% Create mesh
x1_range = [min(X_train_2d(:,1))-1, max(X_train_2d(:,1))+1];
x2_range = [min(X_train_2d(:,2))-1, max(X_train_2d(:,2))+1];

[xx1, xx2] = meshgrid(linspace(x1_range(1), x1_range(2), 200), ...
                       linspace(x2_range(1), x2_range(2), 200));
Z = predict(dt_2d, [xx1(:), xx2(:)]);
[~, ~, Z_numeric] = unique(Z);
Z_reshaped = reshape(Z_numeric, size(xx1));

subplot(2,2,[1 2]);
contourf(xx1, xx2, Z_reshaped, 'LineColor', 'none');
hold on;
colormap(gca, lines(n_classes));

% Plot training data
unique_train_classes = unique(y_train);
for i = 1:length(unique_train_classes)
    idx = y_train == unique_train_classes(i);
    scatter(X_train_2d(idx, 1), X_train_2d(idx, 2), 30, colors(min(i,n_classes),:), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end

xlabel(sprintf('PC1 (%.1f%% var)', explained(1)));
ylabel(sprintf('PC2 (%.1f%% var)', explained(2)));
title('Decision Boundaries (Training Data)', 'FontWeight', 'bold');
grid on;

% Test data
subplot(2,2,3);
contourf(xx1, xx2, Z_reshaped, 'LineColor', 'none');
hold on;
colormap(gca, lines(n_classes));

for i = 1:length(unique_test_classes)
    idx = y_test == unique_test_classes(i);
    scatter(X_test_2d(idx, 1), X_test_2d(idx, 2), 30, colors(i,:), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end

xlabel(sprintf('PC1 (%.1f%% var)', explained(1)));
ylabel(sprintf('PC2 (%.1f%% var)', explained(2)));
title('Test Data on Boundaries', 'FontWeight', 'bold');
grid on;

% Info panel
subplot(2,2,4);
axis off;

pca_text = {
    'PCA PROJECTION', ...
    '══════════════', ...
    '', ...
    sprintf('PC1: %.2f%% var', explained(1)), ...
    sprintf('PC2: %.2f%% var', explained(2)), ...
    sprintf('Total: %.2f%%', sum(explained(1:2))), ...
    '', ...
    'This is a 2D projection', ...
    sprintf('of %d-dimensional space', size(X_train, 2)), ...
    '', ...
    'Boundaries are axis-aligned', ...
    'due to tree structure'
};

text(0.1, 0.95, pca_text, 'FontSize', 10, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree Boundaries (PCA)', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% VISUALIZATION 10: PER-CLASS PERFORMANCE DASHBOARD
% -------------------------------------------------------------------------

fprintf('10. Creating per-class performance dashboard...\n');

figure('Name', 'Decision Tree - Per-Class Performance', 'Position', [100 100 1400 900]);

TN = sum(dt_cm(:)) - (TP + FP + FN);
sensitivity = TP ./ (TP + FN) * 100;
specificity = TN ./ (TN + FP) * 100;
ppv = TP ./ (TP + FP) * 100;
npv = TN ./ (TN + FN) * 100;

% Handle NaN
sensitivity(isnan(sensitivity)) = 0;
specificity(isnan(specificity)) = 0;
ppv(isnan(ppv)) = 0;
npv(isnan(npv)) = 0;

% Sensitivity & Specificity
subplot(3,3,1);
bar([sensitivity, specificity]);
set(gca, 'XTickLabel', class_labels_cell);
ylabel('Percentage (%)');
title('Sensitivity & Specificity', 'FontWeight', 'bold');
legend('Sensitivity', 'Specificity', 'Location', 'best');
grid on;

% Precision & NPV
subplot(3,3,2);
bar([ppv, npv]);
set(gca, 'XTickLabel', class_labels_cell);
ylabel('Percentage (%)');
title('Precision & NPV', 'FontWeight', 'bold');
legend('Precision', 'NPV', 'Location', 'best');
grid on;

% F1-Score
subplot(3,3,3);
bar(f1_score, 'FaceColor', [0.5 0.8 0.5]);
set(gca, 'XTickLabel', class_labels_cell);
ylabel('F1-Score (%)');
title('F1-Score by Class', 'FontWeight', 'bold');
grid on;

% Classification outcomes
subplot(3,3,4);
bar([TP, FP, FN, TN]);
set(gca, 'XTickLabel', class_labels_cell);
ylabel('Count');
title('Classification Outcomes', 'FontWeight', 'bold');
legend('TP', 'FP', 'FN', 'TN', 'Location', 'best');
grid on;

% Support
subplot(3,3,5);
support = sum(dt_cm, 2);
bar(support, 'FaceColor', [0.8 0.6 0.9]);
set(gca, 'XTickLabel', class_labels_cell);
ylabel('Count');
title('Support (Samples)', 'FontWeight', 'bold');
grid on;

% Per-class accuracy
subplot(3,3,6);
accuracy_per_class = (TP + TN) ./ (TP + TN + FP + FN) * 100;
bar(accuracy_per_class, 'FaceColor', [0.3 0.7 0.9]);
set(gca, 'XTickLabel', class_labels_cell);
ylabel('Accuracy (%)');
title('Per-Class Accuracy', 'FontWeight', 'bold');
ylim([0 100]);
grid on;

% Detailed metrics table
subplot(3,3,[7 8 9]);
axis off;

table_text = {'DETAILED METRICS', '════════════════', ''};
for i = 1:length(class_labels_cell)
    table_text{end+1} = sprintf('CLASS %s:', class_labels_cell{i});
    table_text{end+1} = sprintf('├─ Support:     %d', support(i));
    table_text{end+1} = sprintf('├─ Precision:   %.2f%%', ppv(i));
    table_text{end+1} = sprintf('├─ Recall:      %.2f%%', sensitivity(i));
    table_text{end+1} = sprintf('├─ F1:          %.2f%%', f1_score(i));
    table_text{end+1} = sprintf('├─ Specificity: %.2f%%', specificity(i));
    table_text{end+1} = sprintf('└─ TP:%d FP:%d FN:%d TN:%d', TP(i), FP(i), FN(i), TN(i));
    table_text{end+1} = '';
end
table_text{end+1} = '════════════════';
table_text{end+1} = sprintf('OVERALL: %.2f%%', dt_accuracy);

text(0.05, 0.95, table_text, 'FontSize', 9, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Decision Tree - Comprehensive Per-Class Performance', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('\n✓ All Decision Tree visualizations complete!\n\n');


% -------------------------------------------------------------------------
% Method 2: SVM with Error-Correcting Output Codes (ECOC)
% -------------------------------------------------------------------------

fprintf('Training SVM Classifier (ECOC)...\n');
tic;
svm_model = fitcecoc(X_train, y_train);
svm_train_time = toc;

% Predictions
tic;
y_pred_svm = predict(svm_model, X_test);
svm_test_time = toc;

% Evaluation
svm_accuracy = sum(y_pred_svm == y_test) / length(y_test) * 100;
svm_cm = confusionmat(y_test, y_pred_svm, 'Order', sufficient_classes);

fprintf('  Training time: %.4f seconds\n', svm_train_time);
fprintf('  Testing time:  %.4f seconds\n', svm_test_time);
fprintf('  Accuracy: %.2f%%\n\n', svm_accuracy);

% -------------------------------------------------------------------------
% Method 3: Neural Network (Pattern Recognition)
% -------------------------------------------------------------------------

fprintf('Training Neural Network Classifier...\n');

% Convert labels to categorical
% First convert character arrays to cell arrays, then to categorical
y_train_cat = categorical(cellstr(y_train));
y_test_cat = categorical(cellstr(y_test));
classes = categories(y_train_cat);
num_classes = length(classes);

% One-hot encode labels
y_train_onehot = zeros(length(y_train), num_classes);
y_test_onehot = zeros(length(y_test), num_classes);
for i = 1:num_classes
    y_train_onehot(:, i) = (y_train_cat == classes{i});
    y_test_onehot(:, i) = (y_test_cat == classes{i});
end

% Transpose for patternnet (features x samples)
X_train_nn = X_train';
y_train_nn = y_train_onehot';
X_test_nn = X_test';

% Create and train network
tic;
hidden_layer_size = 10;
nn_model = patternnet(hidden_layer_size);
nn_model.trainParam.showWindow = false;  % Suppress training GUI
rng(42)
nn_model = train(nn_model, X_train_nn, y_train_nn);
nn_train_time = toc;

% Predictions
tic;
y_pred_nn_onehot = nn_model(X_test_nn);
[~, y_pred_nn_idx] = max(y_pred_nn_onehot, [], 1);
y_pred_nn = classes(y_pred_nn_idx)';
y_pred_nn = char(y_pred_nn);
nn_test_time = toc;

% Evaluation
nn_accuracy = sum(y_pred_nn == y_test) / length(y_test) * 100;
nn_cm = confusionmat(y_test, y_pred_nn, 'Order', sufficient_classes);

fprintf('  Training time: %.4f seconds\n', nn_train_time);
fprintf('  Testing time:  %.4f seconds\n', nn_test_time);
fprintf('  Accuracy: %.2f%%\n\n', nn_accuracy);

% -------------------------------------------------------------------------
% Visualize ML Results
% -------------------------------------------------------------------------

% Figure: Confusion Matrices
figure('Name', 'ML Classification - Confusion Matrices', 'Position', [100 100 1400 400]);

% Decision Tree
subplot(1,3,1);
confusionchart(dt_cm, sufficient_classes);
title(sprintf('Decision Tree\nAccuracy: %.2f%%', dt_accuracy));

% SVM
subplot(1,3,2);
confusionchart(svm_cm, sufficient_classes);
title(sprintf('SVM (ECOC)\nAccuracy: %.2f%%', svm_accuracy));

% Neural Network
subplot(1,3,3);
confusionchart(nn_cm, sufficient_classes);
title(sprintf('Neural Network\nAccuracy: %.2f%%', nn_accuracy));

% Figure: Model Comparison
figure('Name', 'ML Model Comparison');

% Accuracy comparison
subplot(2,1,1);
models = {'Decision Tree', 'SVM (ECOC)', 'Neural Network'};
accuracies = [dt_accuracy, svm_accuracy, nn_accuracy];
bar(accuracies);
set(gca, 'XTickLabel', models);
ylabel('Accuracy (%)');
title('Model Accuracy Comparison');
ylim([0 110]);  % Increased from 100 to 110 to give space for labels
grid on;
for i = 1:length(accuracies)
    text(i, accuracies(i)+3, sprintf('%.2f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end

% Training time comparison
subplot(2,1,2);
train_times = [dt_train_time, svm_train_time, nn_train_time];
bar(train_times);
set(gca, 'XTickLabel', models);
ylabel('Time (seconds)');
title('Training Time Comparison');
grid on;

% Figure: Classification examples
figure('Name', 'Classification Examples');
num_examples = min(9, size(X_test, 1));
for i = 1:num_examples
    subplot(3, 3, i);
    plot(beat_time_axis_ms, X_test(i, :), 'c', 'LineWidth', 1);
    hold on;
    xline(0, 'r--', 'R', 'LineWidth', 1);
    hold off;
    
    true_label = y_test(i);
    dt_label = y_pred_tree(i);
    svm_label = y_pred_svm(i);
    nn_label = y_pred_nn(i);
    
    title_str = sprintf('True: %c | DT: %c | SVM: %c | NN: %c', ...
        true_label, dt_label, svm_label, nn_label);
    title(title_str, 'FontSize', 8);
    xlabel('Time (ms)');
    ylabel('mV');
    grid on;
end
sgtitle('Classification Examples - Predictions from All Models');

fprintf('Machine learning classification complete!\n\n');

% -------------------------------------------------------------------------
% PART 6B: UNSUPERVISED BEAT CLUSTERING
% -------------------------------------------------------------------------
% This section performs unsupervised clustering on beat morphologies to
% discover natural groupings without using labels. We then compare the
% discovered clusters to expert annotations.
%
% Methods used:
%   1. PCA for dimensionality reduction and visualization
%   2. K-means clustering
%   3. Hierarchical clustering
%   4. t-SNE for 2D visualization

fprintf('PART 6B: UNSUPERVISED BEAT CLUSTERING\n');
fprintf('-------------------------------------\n\n');

% Use the beat matrix (each column is a beat)
% Transpose so each row is a beat for clustering
beat_data = beat_matrix';  % [num_beats x num_samples]
beat_labels = valid_annotations;  % Expert annotations for comparison

fprintf('Beat data for clustering:\n');
fprintf('  Number of beats: %d\n', size(beat_data, 1));
fprintf('  Samples per beat: %d\n', size(beat_data, 2));
fprintf('  Expert labels available: %s\n\n', unique(beat_labels)');

% -------------------------------------------------------------------------
% Step 1: PCA - Dimensionality Reduction
% -------------------------------------------------------------------------
fprintf('Step 1: Principal Component Analysis (PCA)...\n');

% Standardize data (zero mean, unit variance)
beat_data_standardized = zscore(beat_data);

% Perform PCA
[coeff, score, latent, ~, explained] = pca(beat_data_standardized);

% Determine number of components for 95% variance
cumulative_var = cumsum(explained);
num_components_95 = find(cumulative_var >= 95, 1);

fprintf('  Total components: %d\n', length(explained));
fprintf('  Components for 95%% variance: %d\n', num_components_95);
fprintf('  Variance explained by first 3 PCs: %.1f%%\n\n', sum(explained(1:3)));

% Use first 10 components for clustering (or fewer if not available)
num_pca_components = min(10, num_components_95);
beat_pca = score(:, 1:num_pca_components);

fprintf('  Using %d PCA components for clustering\n\n', num_pca_components);

% Visualize the first 6 principal components (eigenbeats)
figure('Name', 'Principal Components (Eigenbeats)');
for i = 1:6
    subplot(2,3,i);
    plot(beat_time_axis_ms, coeff(:,i), 'b', 'LineWidth', 1.5);
    hold on;
    xline(0, 'r--', 'R-peak');
    hold off;
    xlabel('Time (ms)');
    ylabel('Loading');
    title(sprintf('PC%d (%.1f%% var)', i, explained(i)));
    grid on;
end
sgtitle('Principal Components - ECG Beat Patterns');


% -------------------------------------------------------------------------
% Step 2: K-Means Clustering
% -------------------------------------------------------------------------
fprintf('Step 2: K-Means Clustering...\n');

% Determine optimal number of clusters using elbow method
max_k = min(10, length(unique(beat_labels)) + 3);  % Test up to max_k clusters
silhouette_scores = zeros(max_k - 1, 1);
inertia = zeros(max_k - 1, 1);

for k = 2:max_k
    [idx_temp, ~, sumd] = kmeans(beat_pca, k, 'Replicates', 5, 'MaxIter', 200);
    inertia(k-1) = sum(sumd);
    silhouette_scores(k-1) = mean(silhouette(beat_pca, idx_temp));
end

% Find optimal k using silhouette score
[best_silhouette, best_k_idx] = max(silhouette_scores);
optimal_k = best_k_idx + 1;

fprintf('  Testing k = 2 to %d clusters\n', max_k);
fprintf('  Best silhouette score: %.3f at k = %d\n', best_silhouette, optimal_k);

% Perform final k-means with optimal k
[kmeans_clusters, kmeans_centroids] = kmeans(beat_pca, optimal_k, ...
    'Replicates', 10, 'MaxIter', 500, 'Display', 'off');

fprintf('  Final k-means clustering with k = %d\n', optimal_k);

% Analyze cluster composition (compare to expert labels)
fprintf('\n  Cluster composition (vs expert labels):\n');
for c = 1:optimal_k
    cluster_mask = (kmeans_clusters == c);
    cluster_labels = beat_labels(cluster_mask);
    unique_labels = unique(cluster_labels);
    
    fprintf('    Cluster %d (%d beats): ', c, sum(cluster_mask));
    for j = 1:length(unique_labels)
        count = sum(cluster_labels == unique_labels(j));
        pct = 100 * count / sum(cluster_mask);
        fprintf('%c=%.0f%% ', unique_labels(j), pct);
    end
    fprintf('\n');
end
fprintf('\n');

% -------------------------------------------------------------------------
% Step 3: Hierarchical Clustering
% -------------------------------------------------------------------------
fprintf('Step 3: Hierarchical Clustering...\n');

% Compute linkage using Ward's method
linkage_matrix = linkage(beat_pca, 'ward');

% Cut dendrogram at optimal_k clusters (same as k-means for comparison)
hier_clusters = cluster(linkage_matrix, 'maxclust', optimal_k);

fprintf('  Linkage method: Ward''s minimum variance\n');
fprintf('  Number of clusters: %d\n', optimal_k);

% Compare hierarchical to k-means
agreement = sum(kmeans_clusters == hier_clusters) / length(kmeans_clusters) * 100;
fprintf('  Agreement with k-means: %.1f%% (direct label match)\n', agreement);

% Adjusted Rand Index for better comparison
% (accounts for chance agreement)
ari_kmeans_hier = rand_index_adjusted(kmeans_clusters, hier_clusters);
fprintf('  Adjusted Rand Index (k-means vs hierarchical): %.3f\n\n', ari_kmeans_hier);

% Compare clusters to expert annotations
ari_kmeans_expert = rand_index_adjusted(kmeans_clusters, double(beat_labels));
ari_hier_expert = rand_index_adjusted(hier_clusters, double(beat_labels));
fprintf('  Adjusted Rand Index (k-means vs expert): %.3f\n', ari_kmeans_expert);
fprintf('  Adjusted Rand Index (hierarchical vs expert): %.3f\n\n', ari_hier_expert);

% -------------------------------------------------------------------------
% Step 4: t-SNE Visualization
% -------------------------------------------------------------------------
fprintf('Step 4: t-SNE Visualization...\n');

% Perform t-SNE on PCA-reduced data (faster than raw data)
rng(42);  % For reproducibility
tsne_result = tsne(beat_pca, 'NumDimensions', 2, 'Perplexity', 30, 'Verbose', 0);

fprintf('  t-SNE completed (2D embedding)\n\n');

% -------------------------------------------------------------------------
% Visualize Clustering Results
% -------------------------------------------------------------------------

% Figure: PCA Analysis
figure('Name', 'PCA Analysis', 'Position', [100 100 1400 500]);

% Subplot 1: Variance explained
subplot(1,3,1);
bar(explained(1:min(20, length(explained))), 'FaceColor', [0.3 0.6 0.9]);
hold on;
plot(cumulative_var(1:min(20, length(cumulative_var))), 'r-o', 'LineWidth', 2);
yline(95, 'g--', '95%', 'LineWidth', 1.5);
hold off;
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('PCA Variance Explained');
legend('Individual', 'Cumulative', '95% threshold', 'Location', 'east');
grid on;

% Subplot 2: First 2 PCs colored by expert labels
subplot(1,3,2);
unique_expert_labels = unique(beat_labels);
colors_expert = lines(length(unique_expert_labels));
hold on;
for i = 1:length(unique_expert_labels)
    mask = (beat_labels == unique_expert_labels(i));
    scatter(score(mask, 1), score(mask, 2), 20, colors_expert(i,:), 'filled', 'MarkerFaceAlpha', 0.5);
end
hold off;
xlabel('PC1');
ylabel('PC2');
title('PCA - Expert Labels');
legend(cellstr(unique_expert_labels'), 'Location', 'best');
grid on;

% Subplot 3: First 2 PCs colored by k-means clusters
subplot(1,3,3);
colors_kmeans = lines(optimal_k);
hold on;
for c = 1:optimal_k
    mask = (kmeans_clusters == c);
    scatter(score(mask, 1), score(mask, 2), 20, colors_kmeans(c,:), 'filled', 'MarkerFaceAlpha', 0.5);
end
hold off;
xlabel('PC1');
ylabel('PC2');
title(sprintf('PCA - K-Means Clusters (k=%d)', optimal_k));
legend(arrayfun(@(x) sprintf('Cluster %d', x), 1:optimal_k, 'UniformOutput', false), 'Location', 'best');
grid on;

sgtitle('Principal Component Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% Figure: Clustering Comparison
figure('Name', 'Clustering Comparison', 'Position', [100 100 1400 800]);

% Subplot 1: K-means elbow plot
subplot(2,3,1);
yyaxis left;
plot(2:max_k, inertia, 'b-o', 'LineWidth', 1.5);
ylabel('Inertia (Within-cluster sum of squares)');
yyaxis right;
plot(2:max_k, silhouette_scores, 'r-s', 'LineWidth', 1.5);
ylabel('Silhouette Score');
xlabel('Number of Clusters (k)');
title('K-Means: Elbow Method & Silhouette');
xline(optimal_k, 'g--', sprintf('Optimal k=%d', optimal_k), 'LineWidth', 1.5);
grid on;
legend('Inertia', 'Silhouette', 'Location', 'best');

% Subplot 2: Dendrogram (truncated)
subplot(2,3,2);
dendrogram(linkage_matrix, 30, 'ColorThreshold', linkage_matrix(end-optimal_k+2, 3));
title('Hierarchical Clustering Dendrogram');
xlabel('Beat Index (truncated)');
ylabel('Distance');
grid on;

% Subplot 3: Cluster sizes comparison
subplot(2,3,3);
kmeans_sizes = histcounts(kmeans_clusters, 1:optimal_k+1);
hier_sizes = histcounts(hier_clusters, 1:optimal_k+1);
bar_data = [kmeans_sizes; hier_sizes]';
bar(bar_data);
xlabel('Cluster');
ylabel('Number of Beats');
title('Cluster Size Comparison');
legend('K-Means', 'Hierarchical', 'Location', 'best');
grid on;

% Subplot 4: t-SNE with expert labels
subplot(2,3,4);
hold on;
for i = 1:length(unique_expert_labels)
    mask = (beat_labels == unique_expert_labels(i));
    scatter(tsne_result(mask, 1), tsne_result(mask, 2), 20, colors_expert(i,:), 'filled', 'MarkerFaceAlpha', 0.6);
end
hold off;
xlabel('t-SNE 1');
ylabel('t-SNE 2');
title('t-SNE - Expert Labels');
legend(cellstr(unique_expert_labels'), 'Location', 'best');
grid on;

% Subplot 5: t-SNE with k-means clusters
subplot(2,3,5);
hold on;
for c = 1:optimal_k
    mask = (kmeans_clusters == c);
    scatter(tsne_result(mask, 1), tsne_result(mask, 2), 20, colors_kmeans(c,:), 'filled', 'MarkerFaceAlpha', 0.6);
end
hold off;
xlabel('t-SNE 1');
ylabel('t-SNE 2');
title(sprintf('t-SNE - K-Means Clusters (k=%d)', optimal_k));
legend(arrayfun(@(x) sprintf('Cluster %d', x), 1:optimal_k, 'UniformOutput', false), 'Location', 'best');
grid on;

% Subplot 6: Clustering summary
subplot(2,3,6);
axis off;

cluster_summary_text = {
    'CLUSTERING SUMMARY', ...
    '═══════════════════════════', ...
    '', ...
    sprintf('Number of beats: %d', size(beat_data, 1)), ...
    sprintf('PCA components used: %d', num_pca_components), ...
    sprintf('Optimal clusters (k): %d', optimal_k), ...
    '', ...
    'Silhouette Scores:', ...
    sprintf('  K-Means: %.3f', mean(silhouette(beat_pca, kmeans_clusters))), ...
    sprintf('  Hierarchical: %.3f', mean(silhouette(beat_pca, hier_clusters))), ...
    '', ...
    'Agreement with Expert Labels:', ...
    sprintf('  K-Means ARI: %.3f', ari_kmeans_expert), ...
    sprintf('  Hierarchical ARI: %.3f', ari_hier_expert), ...
    '', ...
    'Note: ARI range is [-1, 1]', ...
    '  1.0 = Perfect agreement', ...
    '  0.0 = Random clustering', ...
    ' <0.0 = Worse than random'
};

text(0.05, 0.95, cluster_summary_text, 'FontSize', 9, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Unsupervised Beat Clustering Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% Figure: Cluster Average Beats
figure('Name', 'Cluster Average Beats', 'Position', [100 100 1200 600]);

num_subplot_cols = min(optimal_k, 4);
num_subplot_rows = ceil(optimal_k / num_subplot_cols);

for c = 1:optimal_k
    subplot(num_subplot_rows, num_subplot_cols, c);
    
    cluster_beats = beat_matrix(:, kmeans_clusters == c);
    cluster_avg = mean(cluster_beats, 2);
    cluster_std = std(cluster_beats, 0, 2);
    
    % Plot mean ± std
    fill([beat_time_axis_ms, fliplr(beat_time_axis_ms)], ...
         [cluster_avg' + cluster_std', fliplr(cluster_avg' - cluster_std')], ...
         colors_kmeans(c,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold on;
    plot(beat_time_axis_ms, cluster_avg, 'Color', colors_kmeans(c,:), 'LineWidth', 2);
    xline(0, 'k--', 'R', 'LineWidth', 1);
    hold off;
    
    % Get dominant expert label for this cluster
    cluster_labels = beat_labels(kmeans_clusters == c);
    [unique_labels, ~, label_idx] = unique(cluster_labels);
    label_counts = accumarray(label_idx, 1);
    [~, max_idx] = max(label_counts);
    dominant_label = unique_labels(max_idx);
    dominant_pct = 100 * max(label_counts) / sum(label_counts);
    
    title(sprintf('Cluster %d (n=%d)\nDominant: %c (%.0f%%)', ...
        c, size(cluster_beats, 2), dominant_label, dominant_pct));
    xlabel('Time (ms)');
    ylabel('mV');
    grid on;
end

sgtitle('Average Beat Morphology by K-Means Cluster', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Unsupervised beat clustering complete!\n\n');

% -------------------------------------------------------------------------
% PART 7: COMPREHENSIVE SUMMARY
% -------------------------------------------------------------------------

fprintf('PART 7: COMPREHENSIVE ANALYSIS SUMMARY\n');
fprintf('--------------------------------------\n\n');

% Figure: Complete Analysis Summary Dashboard
figure('Name', 'Complete Analysis Summary Dashboard', 'Position', [50 50 1500 900]);

% Panel 1: Preprocessed ECG with R-Peaks (10 Seconds)
subplot(3,4,1);
samples_10s = 10*fs;
plot(tm(1:samples_10s), preprocessed_signal(1:samples_10s), 'c');
hold on;
peaks_10s = R_peak_locations(R_peak_locations <= samples_10s);
plot(tm(peaks_10s), preprocessed_signal(peaks_10s), 'rv', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
hold off;
title('ECG with R-Peaks (10s)');
xlabel('Time (s)'); ylabel('mV');
grid on;

% Panel 2: Average Beat
subplot(3,4,2);
plot(beat_time_axis_ms, average_beat, 'c', 'LineWidth', 1.5);
hold on;
xline(0, 'r--', 'R', 'LineWidth', 1);
hold off;
title('Average Beat (all beats)');
xlabel('Time (ms)'); ylabel('mV');
grid on;

% Panel 3: Beat Overlay
subplot(3,4,3);
plot(beat_time_axis_ms, beat_matrix(:, 1:min(100, valid_beat_count)), 'Color', [0.5 0.5 0.5 0.2]);
hold on;
plot(beat_time_axis_ms, average_beat, 'r', 'LineWidth', 2);
hold off;
title(sprintf('Beat Overlay (%d beats)', min(100, valid_beat_count)));
xlabel('Time (ms)'); ylabel('mV');
grid on;

% Panel 4: RR Tachogram
subplot(3,4,4);
plot(RR_time_axis/60, RR_intervals_ms, 'c', 'LineWidth', 0.5);
title('RR Tachogram');
xlabel('Time (min)'); ylabel('RR (ms)');
grid on;

% Panel 5: Heart Rate Trend (Instantaneous)
subplot(3,4,5);
plot(RR_time_axis/60, heart_rate_bpm, 'Color', [0.2 0.6 0.2], 'LineWidth', 0.5);
hold on;
yline(average_HR_correct, 'r--', 'LineWidth', 1);
hold off;
title('Instantaneous HR');
xlabel('Time (min)'); ylabel('HR (BPM)');
grid on;

% Panel 6: Windowed Heart Rate
subplot(3,4,6);
plot(window_times_valid/60, window_hr_mean, 'c-', 'LineWidth', 1);
hold on;
yline(final_avg_HR_windowed, 'r--', 'LineWidth', 1);
hold off;
title(sprintf('Sliding HR (%ds)', window_duration_sec));
xlabel('Time (min)'); ylabel('HR (BPM)');
grid on;

% Panel 7: Poincaré Plot
subplot(3,4,7);
scatter(RR_n, RR_n1, 3, 'c', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot([min_RR_plot max_RR_plot], [min_RR_plot max_RR_plot], 'r--');
hold off;
title('Poincaré Plot');
xlabel('RR_n (ms)'); ylabel('RR_{n+1} (ms)');
axis equal; grid on;

% Panel 8: RR Distribution
subplot(3,4,8);
histogram(RR_intervals_ms, 30, 'FaceColor', [0.3 0.5 0.8]);
title('RR Distribution');
xlabel('RR (ms)'); ylabel('Count');
grid on;

% Panel 9: HR Distribution
subplot(3,4,9);
histogram(heart_rate_bpm, 30, 'FaceColor', [0.2 0.7 0.3]);
title('HR Distribution');
xlabel('HR (BPM)'); ylabel('Count');
grid on;

% Panel 10: ML Model Accuracies
subplot(3,4,10);
bar(accuracies);
set(gca, 'XTickLabel', {'DT', 'SVM', 'NN'});
ylabel('Accuracy (%)');
title('ML Classification');
ylim([0 100]);
grid on;

% Panel 11: Beat Type Distribution
subplot(3,4,11);
beat_type_counts = zeros(length(sufficient_classes), 1);
for i = 1:length(sufficient_classes)
    beat_type_counts(i) = sum(y == sufficient_classes(i));
end
bar(beat_type_counts);
set(gca, 'XTickLabel', cellstr(sufficient_classes));
ylabel('Count');
title('Beat Types');
grid on;

% Panel 12: Text Summary
subplot(3,4,12);
axis off;

summary_text = {
    'ANALYSIS SUMMARY', ...
    '══════════════════', ...
    sprintf('Record: MIT-BIH 208'), ...
    sprintf('Duration: %.1f min', recording_duration_min), ...
    sprintf('Sampling: %d Hz', fs), ...
    '', ...
    'R-PEAKS (Annotations)', ...
    sprintf('  Beats: %d', num_beats), ...
    sprintf('  Valid: %d', valid_beat_count), ...
    '', ...
    'HEART RATE', ...
    sprintf('  Traditional: %.1f BPM', average_HR_correct), ...
    sprintf('  Sliding Win: %.1f BPM', final_avg_HR_windowed), ...
    '', ...
    'HRV (Time-Domain)', ...
    sprintf('  SDNN: %.1f ms', SDNN_ms), ...
    sprintf('  RMSSD: %.1f ms', RMSSD_ms), ...
    '', ...
    'HRV (Freq-Domain)', ...
    sprintf('  LF/HF: %.2f', LF_HF_ratio), ...
    '', ...
    'ML CLASSIFICATION', ...
    sprintf('  Best: %.1f%%', max(accuracies)), ...
    '', ...
    'CLUSTERING', ...
    sprintf('  k=%d, ARI=%.2f', optimal_k, ari_kmeans_expert)
};

text(0.05, 0.95, summary_text, 'FontSize', 8, 'FontName', 'FixedWidth', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

sgtitle('Complete ECG Analysis Summary - MIT-BIH Record 208', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% Final Text Summary
% -------------------------------------------------------------------------

fprintf('-----------------------------------------------------------\n');
fprintf('                  FINAL ANALYSIS SUMMARY                   \n');
fprintf('-----------------------------------------------------------\n\n');

fprintf('DATA INFORMATION:\n');
fprintf('─────────────────\n');
fprintf('  Source: MIT-BIH Arrhythmia Database, Record 208\n');
fprintf('  Lead: MLII (Modified Limb Lead II)\n');
fprintf('  Sampling Rate: %d Hz\n', fs);
fprintf('  Duration: %.2f minutes (%.1f seconds)\n\n', recording_duration_min, recording_duration_sec);

fprintf('PREPROCESSING:\n');
fprintf('──────────────\n');
fprintf('  1. DC offset removal (mean subtraction)\n');
fprintf('  2. Baseline drift removal (Butterworth HP, 0.5 Hz, order 4)\n');
fprintf('  3. Powerline removal (IIR Notch, 60 Hz, Q=35)\n\n');

fprintf('R-PEAK DETECTION:\n');
fprintf('─────────────────\n');
fprintf('  Method: Expert annotations (rdann)\n');
fprintf('  Source: Cardiologist-verified labels\n');
fprintf('  Total R-peaks: %d\n', num_beats);
fprintf('  Valid beats extracted: %d\n\n', valid_beat_count);

fprintf('BEAT SEGMENTATION:\n');
fprintf('──────────────────\n');
fprintf('  Window: -%.0f to +%.0f ms around R-peak\n', pre_R_sec*1000, post_R_sec*1000);
fprintf('  Total window: %.0f ms (%d samples)\n', (pre_R_sec+post_R_sec)*1000, total_window_samples);
fprintf('  Valid beats: %d\n\n', valid_beat_count);

fprintf('HEART RATE ANALYSIS:\n');
fprintf('────────────────────\n');
fprintf('  Overall Average: %.1f BPM (%d beats / %.1f min)\n', average_HR_correct, num_beats, recording_duration_min);
fprintf('  Median HR: %.1f BPM\n', median_HR);
fprintf('  HR Range: %.1f - %.1f BPM\n', min_HR, max_HR);
fprintf('  \n');
fprintf('  Sliding Window Analysis (%d-sec windows, %d-sec step):\n', window_duration_sec, window_step_sec);
fprintf('    Total windows: %d\n', num_windows);
fprintf('    Valid windows: %d\n', num_valid_windows);
fprintf('    Final Avg HR (sliding): %.1f BPM\n', final_avg_HR_windowed);
fprintf('    Window HR range: %.1f - %.1f BPM\n\n', window_hr_min, window_hr_max);

fprintf('HRV METRICS:\n');
fprintf('────────────\n');
fprintf('  RR intervals analyzed: %d (removed %d outliers)\n', num_RR, total_outliers);
fprintf('  \n');
fprintf('  Time-Domain:\n');
fprintf('    Mean RR: %.1f ms\n', mean_RR_ms);
fprintf('    SDNN: %.1f ms\n', SDNN_ms);
fprintf('    RMSSD: %.1f ms\n', RMSSD_ms);
fprintf('    pNN50: %.1f%% (%d intervals)\n', pNN50, NN50_count);
fprintf('  \n');
fprintf('  Frequency-Domain:\n');
fprintf('    VLF Power: %.2f ms^2\n', VLF_power);
fprintf('    LF Power:  %.2f ms^2\n', LF_power);
fprintf('    HF Power:  %.2f ms^2\n', HF_power);
fprintf('    LF/HF Ratio: %.2f\n', LF_HF_ratio);
fprintf('    LF (n.u.): %.1f%%\n', LF_norm);
fprintf('    HF (n.u.): %.1f%%\n\n', HF_norm);

fprintf('MACHINE LEARNING CLASSIFICATION:\n');
fprintf('─────────────────────────────────\n');
fprintf('  Classes: %s\n', sufficient_classes');
fprintf('  Training set: %d samples\n', size(X_train, 1));
fprintf('  Testing set: %d samples\n', size(X_test, 1));
fprintf('  \n');
fprintf('  Model Performance:\n');
fprintf('    Decision Tree:    %.2f%% (Train: %.3fs, Test: %.3fs)\n', dt_accuracy, dt_train_time, dt_test_time);
fprintf('    SVM (ECOC):       %.2f%% (Train: %.3fs, Test: %.3fs)\n', svm_accuracy, svm_train_time, svm_test_time);
fprintf('    Neural Network:   %.2f%% (Train: %.3fs, Test: %.3fs)\n\n', nn_accuracy, nn_train_time, nn_test_time);

fprintf('UNSUPERVISED CLUSTERING:\n');
fprintf('────────────────────────\n');
fprintf('  PCA components used: %d (for 95%% variance: %d)\n', num_pca_components, num_components_95);
fprintf('  Optimal clusters (k): %d (silhouette: %.3f)\n', optimal_k, best_silhouette);
fprintf('  \n');
fprintf('  Agreement with Expert Labels (Adjusted Rand Index):\n');
fprintf('    K-Means:      %.3f\n', ari_kmeans_expert);
fprintf('    Hierarchical: %.3f\n\n', ari_hier_expert);

fprintf('FIGURES CREATED:\n');
fprintf('────────────────\n');
fprintf('  ✓ Raw ECG and preprocessing visualizations\n');
fprintf('  ✓ Frequency analysis (FFT)\n');
fprintf('  ✓ Filter responses\n');
fprintf('  ✓ R-peak detection with annotations\n');
fprintf('  ✓ Beat segmentation and overlay\n');
fprintf('  ✓ Heart rate trends (instantaneous and sliding window)\n');
fprintf('  ✓ RR interval distributions and Poincaré plot\n');
fprintf('  ✓ Frequency-domain HRV analysis\n');
fprintf('  ✓ ML confusion matrices and comparisons\n');
fprintf('  ✓ Unsupervised clustering (PCA, t-SNE, k-means, hierarchical)\n');
fprintf('  ✓ Complete analysis summary dashboard\n\n');

fprintf('-----------------------------------------------------------\n');
fprintf('                    ANALYSIS COMPLETE                      \n');
fprintf('-----------------------------------------------------------\n');

% -------------------------------------------------------------------------
% HELPER FUNCTIONS
% -------------------------------------------------------------------------

function ari = rand_index_adjusted(labels1, labels2)
% RAND_INDEX_ADJUSTED Compute the Adjusted Rand Index between two clusterings
%
% The Adjusted Rand Index (ARI) is a measure of similarity between two
% clusterings, adjusted for chance. It ranges from -1 to 1, where:
%   1.0  = Perfect agreement
%   0.0  = Random clustering (agreement by chance)
%   <0.0 = Less agreement than expected by chance
%
% Input:
%   labels1, labels2 - Cluster assignments (numeric or char vectors)
%
% Output:
%   ari - Adjusted Rand Index

    % Convert to numeric if needed
    if ischar(labels1)
        [~, ~, labels1] = unique(labels1);
    end
    if ischar(labels2)
        [~, ~, labels2] = unique(labels2);
    end
    
    labels1 = labels1(:);
    labels2 = labels2(:);
    
    n = length(labels1);
    
    % Get unique labels
    unique1 = unique(labels1);
    unique2 = unique(labels2);
    
    % Build contingency table
    contingency = zeros(length(unique1), length(unique2));
    for i = 1:length(unique1)
        for j = 1:length(unique2)
            contingency(i,j) = sum((labels1 == unique1(i)) & (labels2 == unique2(j)));
        end
    end
    
    % Row and column sums
    a = sum(contingency, 2);  % Row sums
    b = sum(contingency, 1);  % Column sums
    
    % Calculate index components using combinations
    % Sum of C(n_ij, 2) for all cells
    sum_comb_nij = sum(contingency(:) .* (contingency(:) - 1) / 2);
    
    % Sum of C(a_i, 2) for all rows
    sum_comb_a = sum(a .* (a - 1) / 2);
    
    % Sum of C(b_j, 2) for all columns
    sum_comb_b = sum(b .* (b - 1) / 2);
    
    % C(n, 2)
    comb_n = n * (n - 1) / 2;
    
    % Expected index
    expected_index = (sum_comb_a * sum_comb_b) / comb_n;
    
    % Max index
    max_index = (sum_comb_a + sum_comb_b) / 2;
    
    % Adjusted Rand Index
    if max_index == expected_index
        ari = 1;  % Perfect agreement when both are 0
    else
        ari = (sum_comb_nij - expected_index) / (max_index - expected_index);
    end
end
