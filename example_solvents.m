% Import the file
solvents = importdata('datasets/solvents.csv', ',', 1);

X = block(solvents.data);
X.add_labels(2, solvents.textdata(1,2:end));
X.add_labels(1, solvents.textdata(2:end, 1));

pca = lvm({'X', X}, 2);
plot(pca)
