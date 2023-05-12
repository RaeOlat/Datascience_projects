% Grid search
% Random forest hyperparameter tuning. Please wait, this may take a couple
% of minutes to run.
% Randomly splitting the dataset into training and test data
% (80%) = train and (20%) = test

% (80%) = train and (20%) = test
rng(100)
Data_split = cvpartition(size(Heart_Disease,1),'Holdout',0.2);
test_data = Data_split.test;

% Split dataset into training and test data
Heart_Disease_train_split = Heart_Disease(~test_data,:);
Heart_Disease_test_split = Heart_Disease(test_data,:);

% Dividing the data into lables and features split
% Splitting the data into features and labels
Heart_disease_train_labels = Heart_disease_train(:,14);
Heart_disease_train_features = Heart_disease_train(:,1:13);

Heart_disease_test_labels = Heart_disease_test(:,14);
Heart_disease_test_features = Heart_disease_test(:,1:13);

% Creating an array to store the parameters and results of the grid search
metric=[];

% Stating the range of trees, minimum leaves and minimum number of
% predictors
numberOfTrees = [1 20 40 60 80 100];
numberOfLeaves = [1 10 20 30 40 50];
numberOfPredictors = [5 10 15 20 25];
iteration=0;
for Trees = numberOfTrees
    for MinimumLeaf = numberOfLeaves
        for NumberPredicted = numberOfPredictors
            iteration = iteration+1
   RandomTreeForest = TreeBagger(Trees,Heart_disease_train_features,Heart_disease_train_labels,'ClassNames',{'0' '1'},'Prior',[0.78 0.22],'MinLeafSize',MinimumLeaf,'Method','classification','NumPredictorsToSample',NumberPredicted);
   RandomForestTree_result = predict(RandomTreeForest,Heart_disease_test_features);
[PredictedRF, PosteriorRF] = predict(RandomTreeForest,Heart_disease_test_features);

figure()
result = confusionchart(Heart_disease_test_labels.target,Group)

%Calculating perfromance metrics corresponding to grid search parameters

cmt = Random_Forest_Confusion_Matrix'

diagonal = diag(cmt)
sum_of_rows = sum(cmt, 2)

precision = diagonal ./ sum_of_rows
overall_precisionRF = mean(precision)

sum_of_columns = sum(cmt, 1)

recall = diagonal ./ sum_of_columns'
overall_recallRF = mean(recall)

f1_scoreRF = 2*((overall_precisionRF*overall_recallRF)/(overall_precisionRF+overall_recallRF))

Accuracy_RF = 100*sum(diag(cmt))./sum(cmt(:));  
disp(Accuracy_RF)

% Reading from confusion matrix chart:
TP_RF = 26;
FN_RF = 3;
FP_RF = 11;
TN_RF = 20;

specificityRF = (TN_RF./(TN_RF+FP_RF))

%Array to store grid search paramters and results
Metric=[Metric;tree minleaf numpred Accuracy_RF overall_precisionRF overall_recallRF specificityRF f1_scoreRF;]
       end
   end
end

%determining the maximum values for each row to determine the best
%performance metric
maxValues=max(Metric);
optimizedValues=[];
fprintf('Hyperparameters for each objective\n')
fprintf('   tree minleaf numpred Accuracy_RF overall_precisionRF overall_recallRF specificityRF f1_scoreRF\n')
forbestAccuracy=Metric(find(Metric(:,1)==maxValues(1)),:)
fprintf('  tree minleaf numpred Accuracy_RF overall_precisionRF overall_recallRF specificityRF f1_scoreRF\n')
forbestPrecision=Metric(find(Metric(:,2)==maxValues(2)),:)
fprintf('   tree minleaf numpred Accuracy_RF overall_precisionRF overall_recallRF specificityRF f1_scoreRF\n')
forbestRecall=Metric(find(Metric(:,3)==maxValues(3)),:)
fprintf('  tree minleaf numpred Accuracy_RF overall_precisionRF overall_recallRF specificityRF f1_scoreRF\n')
forbestSpecificity=Metric(find(Metric(:,4)==maxValues(4)),:)
fprintf('  tree minleaf numpred Accuracy_RF overall_precisionRF overall_recallRF specificityRF f1_scoreRF\n')
forbestF1score=Metric(find(Metric(:,5)==maxValues(5)),:)



