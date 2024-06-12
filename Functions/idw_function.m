% Inverse Distance Weighting Method
% Developer: Marcus Nobrega
% Description:  Interpolate spatial variables
%
% Inputs:
% X0 = vector with x and y projected coordinates [x,y] where observations
% are available
% F0 = value of the observations at points X0
% Xint = Coordinates [xint,yint] where interpolated points are calculated
% p = exponent of the inverse weight function. Default = 2.
% rad = radius of interest [L]. Default = infinity
% L = norm used for distance calculation. Default is the Euclidean norm = 2
%
% Outputs:
% Fint = values at interpolated point Xint.


function Fint = idw_function(X0,F0,Xint,p,rad,L)

%
% Default input parameters
if nargin < 6 % Less than 6 parameters
    L = 2; % 2nd norm
    if nargin < 5 % Less than 5 parameters
        rad = inf; % Radius of influence is infinity
        if nargin < 4 % Less than 4 parameters
            p = 2; % Normal weight
        end
    end
end
% Basic dimensions
N = size(X0,1); % Number of samples or observations
M = size(X0,2); % Number of variables
Q = size(Xint,1); % Number of interpolation points

% Inverse distance weight output
Fint = zeros(Q,1);
for ipos = 1:Q % For all interpolation points

    % Distance matrix
    DeltaX = X0 - repmat(Xint(ipos,:),N,1); % Distance
    DabsL = zeros(size(DeltaX,1),1); % Absolute distance

    for ncol = 1:M
        DabsL = DabsL + abs(DeltaX(:,ncol)).^L;
    end
    Dmat = DabsL.^(1/L);
    Dmat(Dmat==0) = eps;
    Dmat(Dmat>rad) = inf;

    % Weights
    W = 1./(Dmat.^p);

    % Interpolation
    Fint(ipos) = sum(W.*F0)/sum(W);
end
end