% TITLE: Extract EEG data into Matlab Matrix
%
% SUMMARY: This script extracts the EEG data in the '.set' file for
% direct analysis in Matlab
%
% INPUT: Any Subject/Condition file ie: 01cr.set, 02fa.set, etc.
%
% OUTPUT: Returns 'data' and 'labels'. 'data' is a matrix representing the
% data for 2 electrodes over all trials and data points. 'labels' is an
% array containing the necessary information if the saccade is to the left
% or right. A value of '2' is a right saccade, a value of '1' is a left
% saccade.
%
% Made by: Jonny Giordano
% Date: May 21st, 2019

function [data, labels] = extract_data(data_name, experiment)

%Load in file

%Set 'home' to current working direction
home = pwd;

%Use EEGLab to open a .set file and access EEG data
tEEG = pop_loadset('filename',data_name,'filepath', strcat(home, '\data\'));

%Collect data
data = []; %initialize data array
for electrode = [9 20] %Collect data for each electrode
    temp_data = tEEG.data(electrode,:,:);
    data = [data; temp_data];
end


if experiment == "saccade"
    %Collect labels for saccadic direction
    labels = zeros(1, length(tEEG.epoch)); %Create labels array
    for trial = 1:length(tEEG.epoch) %Find number of trials
        for j = 1:3
            result = char(tEEG.epoch(trial).eventtype(j)); %Convert choice to char
            result = str2num(result(3)); %Select 3rd character, convert to int

            if result == 1 || result  == 2 %Check result, make sure it is correct label
                break;
            end
        end
        labels(trial) = result ; %Collect event type of that trial
    end
    
else
    
    %Collect labels for target jump - no jump
    labels = zeros(1, length(tEEG.epoch)); %Create labels array
    for trial = 1:length(tEEG.epoch) %Find number of trials
        index = cell2mat(tEEG.epoch(trial).eventlatency) == 0; %Find event trigger at 0ms
        event = find(index);
        labels(trial) = cell2mat(tEEG.epoch(trial).eventtype(event));
        
        if labels(trial) == 1 || labels(trial) == 2 %Check if the label is a 'correct rejection' or 'false alarm'. This is where the target did NOT moved. All these case are relabeld to 0
            labels (trial) = 0;
        else
            labels(trial) = 1; %If labels are for 'hits' or 'misses', the target moved. Relabel as '1'
        end
        
    end
    
end
        



 


