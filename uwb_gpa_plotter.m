% MATLAB script to read a text file and a CSV file with three columns (x, y, z) and plot the data
% MATLAB script to read a text file and a CSV file with three columns (x, y, z) and plot the data

% Prompt user to select a text file
[file_txt, path_txt] = uigetfile('*.txt', 'Select the text file');
if isequal(file_txt, 0)
    disp('User canceled text file selection.');
    return;
end

% Construct full file path for text file
filename_txt = fullfile(path_txt, file_txt);

% Read the data from the text file
try
    data_txt = readmatrix(filename_txt);
    if size(data_txt, 2) ~= 3
        error('The text file must have exactly three columns (x, y, z).');
    end
catch
    error('Error reading the text file. Ensure it is a valid text file with three numeric columns.');
end

% Prompt user to select a CSV file
[file_csv, path_csv] = uigetfile('*.csv', 'Select the CSV file');
if isequal(file_csv, 0)
    disp('User canceled CSV file selection.');
    return;
end

% Construct full file path for CSV file
filename_csv = fullfile(path_csv, file_csv);

% Read the data from the CSV file
try
    data_csv = readmatrix(filename_csv);
    if size(data_csv, 2) < 10
        error('The CSV file must have at least 10 columns with x in 8th, y in 9th, and z in 10th column.');
    end
catch
    error('Error reading the CSV file. Ensure it is a valid CSV file with at least 10 numeric columns.');
end

% Extract x, y, and z columns for text file
txt_x = data_txt(:, 2);
txt_y = -data_txt(:, 1); % Flip the sign of y
txt_z = data_txt(:, 3);

% Extract x, y, and z columns for CSV file
csv_x = data_csv(:, 8);
csv_y = data_csv(:, 9); % Flip the sign of y
csv_z = data_csv(:, 10);

% Adjust time vectors for different sampling rates
num_txt_samples = length(txt_x);
txt_time = linspace(0, num_txt_samples / 10, num_txt_samples); % 10 samples per second for test 1; 9.5 samp/sec for test 2

num_csv_samples = length(csv_x);
csv_time = linspace(0, num_csv_samples * 2, num_csv_samples); % 1 sample every 2 seconds

% Create figure with three subplots
figure('Name', 'NEU Reference Frame for UWB and GPS Positions', 'NumberTitle', 'off');
tiledlayout(3,1);
title('NEU Reference Frame for UWB and GPS Positions: Test 1', 'FontSize', 14, 'FontWeight', 'bold');
% X comparison plot
subplot(3,1,1);
plot(txt_time, txt_x, 'r', 'LineWidth', 1.5);
hold on;
plot(csv_time, csv_x, 'b', 'LineWidth', 1.5);
hold off;
grid on;
ylabel('X-axis (m)');
legend({'UWB Calculated Position', 'GPS Coordinates'});
title('NEU Reference Frame for UWB and GPS Positions: X Position Comparison');

% Y comparison plot
subplot(3,1,2);
plot(txt_time, txt_y, 'r', 'LineWidth', 1.5);
hold on;
plot(csv_time, csv_y, 'b', 'LineWidth', 1.5);
hold off;
grid on;
ylabel('Y-axis (m)');
legend({'UWB Calculated Position', 'GPS Coordinates'});
title('NEU Reference Frame for UWB and GPS Positions: Y Position Comparison');

% Z comparison plot
subplot(3,1,3);
plot(txt_time, txt_z, 'r', 'LineWidth', 1.5);
hold on;
plot(csv_time, csv_z, 'b', 'LineWidth', 1.5);
hold off;
grid on;
%ylabel('Z-axis');
xlabel('Time (seconds)');
ylabel('Z-axis (m)');
legend({'UWB Calculated Position', 'GPS Coordinates'});
title('NEU Reference Frame for UWB and GPS Positions: Z Position Comparison');

disp('Plots generated successfully.');

% 
% % Prompt user to select a text file
% [file_txt, path_txt] = uigetfile('*.txt', 'Select the text file');
% if isequal(file_txt, 0)
%     disp('User canceled text file selection.');
%     return;
% end
% 
% % Construct full file path for text file
% filename_txt = fullfile(path_txt, file_txt);
% 
% % Read the data from the text file
% try
%     data_txt = readmatrix(filename_txt);
%     if size(data_txt, 2) ~= 3
%         error('The text file must have exactly three columns (x, y, z).');
%     end
% catch
%     error('Error reading the text file. Ensure it is a valid text file with three numeric columns.');
% end
% 
% % Prompt user to select a CSV file
% [file_csv, path_csv] = uigetfile('*.csv', 'Select the CSV file');
% if isequal(file_csv, 0)
%     disp('User canceled CSV file selection.');
%     return;
% end
% 
% % Construct full file path for CSV file
% filename_csv = fullfile(path_csv, file_csv);
% 
% % Read the data from the CSV file
% try
%     data_csv = readmatrix(filename_csv);
%     if size(data_csv, 2) < 10
%         error('The CSV file must have at least 10 columns with x in 8th, y in 9th, and z in 10th column.');
%     end
% catch
%     error('Error reading the CSV file. Ensure it is a valid CSV file with at least 10 numeric columns.');
% end
% 
% % Extract x, y, and z columns for text file
% txt_x = data_txt(:, 1);
% txt_y = -data_txt(:, 2); % Flip the sign of y
% txt_z = data_txt(:, 3);
% 
% % Extract x, y, and z columns for CSV file
% csv_x = data_csv(:, 8);
% csv_y = data_csv(:, 9); % Flip the sign of y
% csv_z = data_csv(:, 10);
% 
% % Create figure with three subplots
% figure('Name', 'Test 1', 'NumberTitle', 'off');
% 
% % X comparison plot
% subplot(3,1,1);
% plot(txt_x, 'r', 'LineWidth', 1.5);
% hold on;
% plot(csv_x, 'b', 'LineWidth', 1.5);
% hold off;
% grid on;
% ylabel('X-axis');
% legend({'UWB Calculated Position', 'GPS Coordinates'});
% title('X Position Comparison');
% 
% % Y comparison plot
% subplot(3,1,2);
% plot(txt_y, 'r', 'LineWidth', 1.5);
% hold on;
% plot(csv_y, 'b', 'LineWidth', 1.5);
% hold off;
% grid on;
% ylabel('Y-axis');
% legend({'UWB Calculated Position', 'GPS Coordinates'});
% title('Y Position Comparison');
% 
% % Z comparison plot
% subplot(3,1,3);
% plot(txt_z, 'r', 'LineWidth', 1.5);
% hold on;
% plot(csv_z, 'b', 'LineWidth', 1.5);
% hold off;
% grid on;
% ylabel('Z-axis');
% xlabel('Sample Index');
% legend({'UWB Calculated Position', 'GPS Coordinates'});
% title('Z Position Comparison');
% 
% disp('Plots generated successfully.');
