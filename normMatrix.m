load labelMatrix.mat;
wordsNum = size(labelMatrix, 1);
for i = 1 : wordsNum
    rowNorm = norm(labelMatrix(i, :));
    if rowNorm
        labelMatrix(i, :) = labelMatrix(i, :) / rowNorm;
    else
        fprintf('%d-th row is 0-vector\n', i);
    end
end
save labelMatrix_pro.mat labelMatrix;
