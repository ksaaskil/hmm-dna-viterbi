function MAP_path=sequence_viterbi(observations)
% MAP_path=sequence_viterbi(observations) returns the most probable 
% chain of H's and L's as a string,  given the observations as an
% input (string with letters A, G, C, T).
% 
% The function defines the following parameters, which are fed into 
% the Viterbi algorithm (viterbi.m):
% - Likelihoods of observing A, C, G, or T in states H and L (matrix Q)
% - Transition matrix for transition probabilities between states H and L (matrix T)
% - Prior probablities for the states H and L (vector P0)
%
% These parameters should be modified accordingly when changing the model. 

% Allow for lowercase letters as input by changing letters to uppercase
observations=upper(observations);

% Check that only letters A, C, G, T are included in the input
if regexp(observations,'[^ACGT]','once')
    error('Only letters A, C, G, and T are allowed in the sequence! Your input was %s.\n',observations);
end

% Transition matrix T
% First column includes transition probabilities for H->H and H->L,
% Second column includes probabilitites for L->H and L->L
T=[0.5,0.4;
    0.5,0.6];

% Likelihood matrix Q
% The matrix defines the probabilities of observing A, C, G, and T in H and L states
% First row corresponds to the H state
% Second row corresponds to the L state
Q=[0.2,0.3,0.3,0.2;
    0.3,0.2,0.2,0.3];

% Prior probabilities for H and L states
P0=[0.5;
    0.5];


% Encode the observations into a vector of integers ranging from 1 to 4
y=zeros(length(observations),1);
y(observations=='A')=1;
y(observations=='C')=2;
y(observations=='G')=3;
y(observations=='T')=4;

% More general mapping could be obtained using key-value pairs
% keySet =   {'A', 'C', 'G', 'T'};
% valueSet = {1,2,3,4};
% mapObj = containers.Map(keySet,valueSet);

% The number of states (H and L)
Ns=2;

% The number of observations
N=length(y);

% Feed the parameters to the Viterbi algorithm
MAP_path=viterbi(T,Q,P0,y,Ns,N);

% Turn the output into an array of characters ('0' or '1')
MAP_path=int2str(MAP_path);
% Turn integers into 'H' and 'L';
MAP_path(MAP_path=='1')='H';
MAP_path(MAP_path=='2')='L';
% Concatenate into a single string with no spaces
MAP_path=strjoin(cellstr(MAP_path)','');

% For output
% fprintf('The most probable path is %s.\n',MAP_path);
