clc
clear

train = load('-mat', 'train_32x32');
test = load('-mat', 'test_32x32');

size(train.X)
train.X = permute(train.X, [4 3 2 1]);
size(train.X)
train.X = reshape(train.X, [], 32*32*3);

size(test.X)
test.X = permute(test.X, [4 3 2 1]);
size(test.X)
test.X = reshape(test.X, [], 32*32*3);

CVNAME = ['A', 'B', 'C', 'D', 'E'];

y = [train.y; test.y];
y(y==10) = 0;

X = [train.X; test.X];

data = [y X];

cv = cvpartition(y,'KFold',5);
for i = 1:5
    idx = cv.test(i);
    hist(y(idx));
    
    csvwrite([CVNAME(i) '.txt'], data(idx, :))
end
