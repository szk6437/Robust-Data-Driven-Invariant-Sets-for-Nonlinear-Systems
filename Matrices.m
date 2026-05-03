% Clear workspace and command window for a clean start
clear;
clc;

% --- Read data from CSV files ---

% Define the filenames
m_filename = 'M.csv';
n_filename = 'N.csv';

% Read the data from the CSV files into matrices
% 'readmatrix' is the recommended function for reading numeric data from text files
M = readmatrix(m_filename);
N = readmatrix(n_filename);

% --- Verification ---

% Display a message to confirm the process is complete
disp('Successfully loaded M.csv and N.csv into the workspace.');

% Display the size of the loaded matrices to verify they are correct
disp('Size of matrix M:');
disp(size(M));

disp('Size of matrix N:');
disp(size(N));

% This script assumes M and N are already in your MATLAB workspace.

% --- Call con2vert to find the vertices ---
disp('Calling con2vert to find the vertices of the polytope M*x <= N...');
try
    % The main function call:
    V = con2vert(M, N);
    
    % --- Display Results ---
    [num_vertices, num_variables] = size(V);
    fprintf('Successfully found %d vertices.\n', num_vertices);
    fprintf('The vertex matrix V has a size of %d x %d.\n', num_vertices, num_variables);
    
    % --- Save the V matrix to a CSV file ---
    writematrix(V, 'V.csv');
    fprintf('Successfully saved the vertex matrix to V.csv\n\n'); % Added confirmation
    
    % Display the first few vertices if available
    if num_vertices > 5
        disp('First 5 vertices:');
        disp(V(1:5, :));
    else
        disp('Vertices:');
        disp(V);
    end
    
catch ME
    % Catch potential errors from con2vert
    fprintf('An error occurred during vertex computation: %s\n', ME.message);
    disp('This often happens if the constraints do not form a bounded region.');
    disp('Please check the notes in the con2vert.m file for more details.');
end