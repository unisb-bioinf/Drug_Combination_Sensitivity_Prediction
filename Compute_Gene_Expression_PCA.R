library(data.table)

# get arguments from json file passed as argument (see Example_PCA.json)
args = commandArgs(trailingOnly=TRUE)
library(jsonlite)
json_data = fromJSON(args[1])

# create variables from json fields
for(i in 1:length(json_data)) {
  assign(names(json_data)[[i]], json_data[[i]])
}

# COSMIC IDs or names of train/test cell lines
cells_train = read.csv(train_path, sep = "\t", header = F)
cells_test = read.csv(test_path, sep = "\t", header = F)

# gene expression (rows: cell lines, cols: genes), matrix can be obtained from the GDSC website: https://www.cancerrxgene.org/downloads/bulk_download
exp = data.frame(fread(exp_path, check.names = F, stringsAsFactors = F))
rownames(exp) = exp$V1
exp$V1 = NULL

train_data = exp[rownames(exp) %in% cells_train$V1,]
test_data = exp[rownames(exp) %in% cells_test$V1,]

pca = prcomp(train_data, center = TRUE, scale. = TRUE, rank. = num_features)

features_train = data.frame(predict(pca, newdata = train_data))
features_test = data.frame(predict(pca, newdata = test_data))
rownames(features_train) = cells_train$V1
rownames(features_test) = cells_test$V1

all_pca = rbind(features_train, features_test)

write.table(all_pca, output_file, sep = "\t", row.names = T, col.names = T, quote = F)


