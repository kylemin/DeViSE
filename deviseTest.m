%load ../fc7_val.mat
%load /mnt/brain3/datasets/extra/alex_features/alex_fc7_val.mat
%vdata = data;
%vlabels = label;
%load /mnt/brain3/datasets/extra/imagenet2012/vggnet_fc7_train.mat
%load /mnt/brain3/datasets/extra/alex_features/alex_fc7_train.mat
%labels = label;
%load /mnt/brain2/scratch/kibok/private-homedir/taxonomy_v2.1/taxonomy/taxonomy_full_ilsvrc2012.mat

load ./vec/labelMatrix_500_1.mat;
%labelMatrix = labelMatrix(1:1000, :);

fout = fopen('normResult.txt', 'w');
wordsNum = size(labelMatrix, 1);
zeroLabels = [];
for i = 1 : wordsNum
    rowNorm = norm(labelMatrix(i, :));
    if rowNorm
        labelMatrix(i, :) = labelMatrix(i, :) / rowNorm;
        %labelMatrix(i, :) = labelMatrix(i, :);
    else
        fprintf('%d-th row is 0-vector\n', i);
        fprintf(fout, '%d-th row is 0-vector\n', i);
        zeroLabels = [zeroLabels; i];
    end
end
fclose(fout);

rng(1,'twister');

wordsNum = 1000;
for i = 1 : wordsNum
    rowNorm = norm(labelMatrix(i, :));
    startDist = 1;
    bCondition = true;
    if ~rowNorm
        allParents = ancestors{i};
        parents_hop = ancestors_hop{i};
        while bCondition
            parents = allParents(parents_hop == startDist);
            parIdx = randperm(length(parents));
            for j = 1 : length(parIdx)
                parVec = labelMatrix(parents(parIdx(j)), :);
                if norm(parVec)
                    parVec = parVec + randn(1, size(labelMatrix, 2))*1e-1;
                    labelMatrix(i, :) = parVec / norm(parVec);
                    fprintf('%d-th row is filled up\n', i);
                    bCondition = false;
                    break;
                end
            end
            startDist = startDist + 1;
        end
    end
end

[n, m] = size(data);
mv = size(vdata, 2);
[labelNum, emDim] = size(labelMatrix);

rng(0,'twister');
w = randn(emDim, n)*1e-2;
velocity = zeros(size(w));
mgin = 0.1;

epochs = 10;
minibatch = 5000;
alpha = 1e-1;
momentum = 0.9;
mom = 0.5;
momIncrease = 20;
hitDist = 2;
weightsPath = './weights/';
if ~exist(weightsPath, 'dir')
    mkdir(weightsPath);
end

fid = fopen('result.txt','wt');
numIter = floor(m/minibatch);
tic;
for e = 1:epochs
    if e > 0
        correct = 0;
        correct2 = 0;
        for i=1:minibatch:(mv-minibatch+1)
            tmp = 0;
            f = w*max([vdata(:,i:i+minibatch-1); ones(1, minibatch)], 0);
            f = labelMatrix*f;
            [~, label] = max(f(1:1000, :), [], 1);
            correct = correct + sum(vlabels(i:i+minibatch-1) == label');
            for j = 1 : minibatch
                if taxonomy_dist_mat(label(j), vlabels(i+j-1)) <= hitDist
                    tmp = tmp + 1;
                end
            end
            correct2 = correct2 + tmp;
        end
        accuracy = correct / mv;
        accuracy2 = correct2 / mv;
    end
    
    tEpochS = tic;
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    it = 0;
    
    for s=1:minibatch:(m-minibatch+1)
        tIterS = tic;
        it = it + 1;
        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = momentum;
        elseif it == floor(numIter/3)
            alpha = max(alpha/1.26, 1e-8);
            correct = 0;
            correct2 = 0;
            for i=1:minibatch:(mv-minibatch+1)
                tmp = 0;
                f = w*max([vdata(:,i:i+minibatch-1); ones(1, minibatch)], 0);
                f = labelMatrix*f;
                [~, label] = max(f(1:1000, :), [], 1);
                correct = correct + sum(vlabels(i:i+minibatch-1) == label');
                for j = 1 : minibatch
                    if taxonomy_dist_mat(label(j), vlabels(i+j-1)) <= hitDist
                        tmp = tmp + 1;
                    end
                end
                correct2 = correct2 + tmp;
            end
            accuracy = correct / mv;
            accuracy2 = correct2 / mv;
            %save(strcat('weights_third_',num2str(e)),'w');
        elseif it == floor(numIter/3)*2
            alpha = max(alpha/1.26, 1e-8);
            correct = 0;
            correct2 = 0;
            for i=1:minibatch:(mv-minibatch+1)
                tmp = 0;
                f = w*max([vdata(:,i:i+minibatch-1); ones(1, minibatch)], 0);
                f = labelMatrix*f;
                [~, label] = max(f(1:1000, :), [], 1);
                correct = correct + sum(vlabels(i:i+minibatch-1) == label');
                for j = 1 : minibatch
                    if taxonomy_dist_mat(label(j), vlabels(i+j-1)) <= hitDist
                        tmp = tmp + 1;
                    end
                end
                correct2 = correct2 + tmp;
            end
            accuracy = correct / mv;
            accuracy2 = correct2 / mv;
            %save(strcat('weights_twothird_',num2str(e)),'w');
        end;
        % get next randomly selected minibatch
        mb_x = max([data(:, rp(s:s+minibatch-1)); ones(1, minibatch)], 0);
        mb_labels = labels(rp(s:s+minibatch-1));
        
        % evaluate the objective function on the next minibatch
        t = labelMatrix(mb_labels', :);
        [cost, grad] = deviseLoss(w, labelMatrix, t, mb_labels, mb_x, mgin);
        velocity = mom * velocity + alpha * grad;
        w = w - velocity;
        
        tIterEnd = toc(tIterS);
        nIt = (e-1)*numIter+it;
        fprintf('Epoch %4d: Cost on iteration %8d = %8.4f, acc = %6.4f, acc2 = %6.4f ',e,nIt,cost,accuracy,accuracy2);
        fprintf('tooks %.2f seconds.\n', tIterEnd);
        fprintf(fid,'Epoch %4d: Cost on iteration %8d = %8.4f, acc = %6.4f, acc2 = %6.4f ',e,nIt,cost,accuracy,accuracy2);
        fprintf(fid,'tooks %.2f seconds.\n', tIterEnd);
    end;
    
    % aneal learning rate by factor of two after each epoch
    alpha = max(alpha/1.26, 1e-8);
    tEpochEnd = toc(tEpochS);
    fprintf('Epoch tooks %.2f seconds.\n', tEpochEnd);
    fprintf(fid,'Epoch tooks %.2f seconds.\n', tEpochEnd);
    save([weightsPath strcat('weights_',num2str(e))], 'w');
end;

tEnd = toc;
fprintf('Training tooks %.2f seconds.\n', tEnd);
fprintf(fid,'Training tooks %.2f seconds.\n', tEnd);

correct = 0;
correct2 = 0;
for i=1:minibatch:(mv-minibatch+1)
    tmp = 0;
    f = w*max([vdata(:,i:i+minibatch-1); ones(1, minibatch)], 0);
    f = labelMatrix*f;
    [~, label] = max(f(1:1000, :), [], 1);
    correct = correct + sum(vlabels(i:i+minibatch-1) == label');
    for j = 1 : minibatch
        if taxonomy_dist_mat(label(j), vlabels(i+j-1)) <= hitDist
            tmp = tmp + 1;
        end
    end
    correct2 = correct2 + tmp;
end
accuracy = correct / mv;
accuracy2 = correct2 / mv;

fprintf('Training acc = %6.4f, acc2 = %6.4f\n',accuracy,accuracy2);
fprintf(fid,'Training acc = %6.4f, acc2 = %6.4f\n',accuracy,accuracy2);

fclose(fid);

