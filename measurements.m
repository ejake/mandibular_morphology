%% Load File
clear all;
n = 229; %Number of examples
n = 1;
%M = dlmread('/home/ajaque/Documents/PhD/mandibular_morphology/data/BDDCoordenadasPerfil29_05_2012.csv', ',',1,1);
M = dlmread('/home/ajaque/Documents/PhD/mandibular_morphology/data/type_landmarks.csv', ',',1,1);
id = 1:n;
MI = [id' M];
%lm = 4;%lm = 9;
lm = size(MI,2);

%% Metrics - lines
% 6 lines per example (combine 2 of 6)
for i=1:n
    vma = MI(i,2:lm);
    r = [vma([1:2:length(vma)])' vma([2:2:length(vma)])'];
    d = lines(r, length(vma)/2);
    if i==1
        ML = [id(i) d];
    else
        ML = [ML; id(i) d];
    end
end
%% Write lines in file
dlmwrite('/home/ajaque/Documents/PhD/mandibular_morphology/data/Perfil_Lines_type.csv', ML, 'delimiter', ',', 'precision', '%d');

%% Metrics - Angles
% 4 triangles per example (combine 3 of 6), 12 angles
for i=1:n
    vma = MI(i,2:lm);
    r = [vma([1:2:length(vma)])' vma([2:2:length(vma)])'];
    an = angles(r, length(vma)/2);
    if i==1
        MA = [id(i) an];
    else
        MA = [MA; id(i) an];
    end
end

%% Write lines in file
dlmwrite('/home/ajaque/Documents/PhD/mandibular_morphology/data/Perfil_Angles_type.csv', MA, 'delimiter', ',', 'precision', '%d');
