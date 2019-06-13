% TITLE: Extract EEG data into Matlab Matrix
%
% SUMMARY: This script extracts the EEG data in the '.set' file for
% direct analysis in Matlab
%
% INPUT: Any Subject/Condition file ie: 01cr.set, 02fa.set, etc.
%
% OUTPUT: 
%
% Made by: Jonny Giordano
% Data: May 21st, 2019

function [data, labels] = extract_data_basic(data_name)

%Load in file

%Set 'home' to current working direction
home = pwd;

%Choose file to access, this must be a cell array so single quotes only
data_name = '01cr.set';

%Use EEGLab to open a .set file and access EEG data
tEEG = pop_loadset('filename',data_name,'filepath', strcat(home, '\data\'));

%Collect data

%Collect data for all trials from left electrode, left HEOG is 9
Left = tEEG.data(9,:,:);

%Collect data from all trials from right electrode, right HEOG is 20
Right = tEEG.data(20,:,:);

%Concatenate data

data = [Left; Right];

    
%Collect labels for each trials
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
    
    labels = labels'
    
end


end
        



