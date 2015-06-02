function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K to the number of centroids
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Separate out the centroids and examples
numCentroids = K;
numExamples = size(X, 1);

% Outer loop over examples
for idxExample = 1:numExamples
    
    % Create an array to store all the norms of the current
    % example to the centroids
    exampleNorms = zeros(numCentroids, 1);
    
    % Calculate norms to all centroids from current example
    for idxCentroid = 1:numCentroids        
        exampleNorms(idxCentroid) = norm(X(idxExample,:) - centroids(idxCentroid,:), 2);    
    end
    
    % Now find the smallest norm (closest centroid to example) and
    % store in the idx to be returned
    [val, index] = min(exampleNorms);
    idx(idxExample) = index;
    
end






% =============================================================

end

