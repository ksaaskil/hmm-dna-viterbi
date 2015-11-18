function MAP_path=viterbi(T,Q,P0,y,Ns,N)
% MAP_path=viterbi(T,Q,P0,y,Ns,N) return the most probable path
% MAP_path of hidden states, given the transition matrix T, observation
% likelihoods Q, prior probabilities P0, vector of observations y,
% number of hidden states Ns in the model, and the number of 
% observations N. The inputs must satisfy the following:
%
% T is a square matrix of size Ns 
% Q is a matrix of size Ns*No, where the number of possible observations
% No must be larger than max(y)
% P0 must be a column vector of size Ns*1
% y must be a vector of length N
%
% MAP_path is a column vector of integers corresponding to the 
% most probable path

% Check that Ns and N are scalars
validateattributes(Ns,{'numeric'},{'scalar'},'viterbi','Ns',5);
validateattributes(N,{'numeric'},{'scalar'},'viterbi','N',6);

% Check that T is a matrix of size Ns times Ns
validateattributes(T,{'numeric'},{'size',[Ns,Ns]},'viterbi','T',1);

% Check that Q has Ns rows
validateattributes(Q,{'numeric'},{'nrows',Ns},'viterbi','Q',2);

% Check that Q has sufficiently many columns (for each possible observation)
if (size(Q,2)<max(y))
   error('Viterbi: Size of Q must be Ns*No, where No>=max(y)!');
end

% Check that P0 is a vector of length Ns
validateattributes(P0,{'numeric'},{'vector','numel',Ns},'viterbi','P0',3);

% Check that y is a vector of length N
validateattributes(y,{'numeric'},{'vector','numel',N},'viterbi','y',4);

% Initialize
V=zeros(Ns,N);
B=zeros(Ns,N);

% Logarithms of likelihoods and transition probabilities, avoiding
% negative Infs (which do not actually hurt)
log_Q=log(Q+1e-300);
log_T=log(T+1e-300);

% First step
% Vector of log-probabilities depending on the prior and the likelihood 
% of first observation
V(:,1)=log(P0+1e-300); % Allows P0 to be a column or row vector 
V(:,1)=V(:,1)+log_Q(:,y(1)); % Add likelihoods

% Vector of current hidden states
B(:,1)=1:Ns;

% Iterate over all observations, updating costs and highest-probability states
for t=2:N
    % Iterate over all states, looking for the previous state that gives the 
    % highest log-probability
    for i=1:Ns
        % Determine the cumulative cost for arriving from each previous state
        % according to the Viterbi algorithm, using the observation y(t)
        % in the likelihood
        costs=V(:,t-1)+log_T(:,i)+log_Q(i,y(t));
        % Choose the maximum index and the value
        [maxval,maxind]=max(costs);
        % Update V and B
        V(i,t)=maxval;
        B(i,t)=maxind;
    end
end

% Find the state with highest log-probability at the end 
[~,finalstate]=max(V(:,N));

% Backtracking
MAP_path=zeros(N,1);
MAP_path(N)=finalstate;
for t=N-1:-1:1
   MAP_path(t)=B(MAP_path(t+1),t+1); 
end
