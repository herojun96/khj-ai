# khj-ai

net = torchvision.models.resnet18(pretrained = True)
resnet18로 모델을 설정해준다.





net.fc = nn.Linear(net.fc.in_features,3)
net = net.to(0)
resnet18은 마지막 linear단계에서 out_feature의 수가 1000이므로 3으로 수정해준다.





criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.0001)
#optimizer = optim.SGD(net.parameters(), lr=0.001)
Adam optimizer가 너 좋은 성능을 보이므로 Adam사용 






    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(0), labels.to(0)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        temp += (predicted == labels).sum().item()
        total += inputs.shape[0]
"
train dateset을 통해서 모델을 훈련시키는 부분







    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(0), labels.to(0)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('test셋 결과물: %d %%' % (
        100 * correct / total))

test dataset을 통해서 정확도를 검사하는 부분





    mat = torch.zeros((3,3))
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(0), labels.to(0)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted, labels)
            c = (predicted == labels).squeeze()
            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            for i,x in enumerate(labels):
              mat[predicted[i],x] += 1
    print(mat)
    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
normal, covid, viral pneumonia 별로 정확도를 측정하고 tesor형식으로 이미지들을 어떤 label로 판별했는지 보여주는 



결과
[1,     1] trainloss: 0.512, trainaccuracy : 0.50
[1,     3] trainloss: 1.112, trainaccuracy : 0.34
[1,     5] trainloss: 0.556, trainaccuracy : 0.81
[1,     7] trainloss: 0.485, trainaccuracy : 0.88
[1,     9] trainloss: 0.388, trainaccuracy : 0.84
[1,    11] trainloss: 0.263, trainaccuracy : 0.94
[1,    13] trainloss: 0.179, trainaccuracy : 0.97
[1,    15] trainloss: 0.175, trainaccuracy : 0.97
[1,     1] testloss: 0.610, testaccuracy : 0.74
[1,     3] testloss: 0.928, testaccuracy : 0.56
[1,     5] testloss: 1.622, testaccuracy : 0.28
[2,     1] trainloss: 0.090, trainaccuracy : 0.94
[2,     3] trainloss: 0.130, trainaccuracy : 1.00
[2,     5] trainloss: 0.141, trainaccuracy : 0.94
[2,     7] trainloss: 0.103, trainaccuracy : 1.00
[2,     9] trainloss: 0.230, trainaccuracy : 0.88
[2,    11] trainloss: 0.077, trainaccuracy : 0.97
[2,    13] trainloss: 0.258, trainaccuracy : 0.88
[2,    15] trainloss: 0.202, trainaccuracy : 0.94
[2,     1] testloss: 0.746, testaccuracy : 0.67
[2,     3] testloss: 0.569, testaccuracy : 0.78
[2,     5] testloss: 1.839, testaccuracy : 0.22
[3,     1] trainloss: 0.070, trainaccuracy : 0.94
[3,     3] trainloss: 0.076, trainaccuracy : 0.97
[3,     5] trainloss: 0.033, trainaccuracy : 1.00
[3,     7] trainloss: 0.058, trainaccuracy : 0.97
[3,     9] trainloss: 0.156, trainaccuracy : 0.94
[3,    11] trainloss: 0.022, trainaccuracy : 1.00
[3,    13] trainloss: 0.080, trainaccuracy : 1.00
[3,    15] trainloss: 0.064, trainaccuracy : 1.00
[3,     1] testloss: 0.847, testaccuracy : 0.63
[3,     3] testloss: 0.836, testaccuracy : 0.62
[3,     5] testloss: 1.784, testaccuracy : 0.28
[4,     1] trainloss: 0.056, trainaccuracy : 0.94
[4,     3] trainloss: 0.095, trainaccuracy : 0.94
[4,     5] trainloss: 0.128, trainaccuracy : 0.97
[4,     7] trainloss: 0.095, trainaccuracy : 0.97
[4,     9] trainloss: 0.047, trainaccuracy : 0.97
[4,    11] trainloss: 0.065, trainaccuracy : 1.00
[4,    13] trainloss: 0.024, trainaccuracy : 1.00
[4,    15] trainloss: 0.141, trainaccuracy : 0.97
[4,     1] testloss: 0.798, testaccuracy : 0.59
[4,     3] testloss: 0.579, testaccuracy : 0.72
[4,     5] testloss: 1.864, testaccuracy : 0.17
[5,     1] trainloss: 0.009, trainaccuracy : 1.00
[5,     3] trainloss: 0.026, trainaccuracy : 1.00
[5,     5] trainloss: 0.018, trainaccuracy : 1.00
[5,     7] trainloss: 0.012, trainaccuracy : 1.00
[5,     9] trainloss: 0.034, trainaccuracy : 1.00
[5,    11] trainloss: 0.046, trainaccuracy : 1.00
[5,    13] trainloss: 0.027, trainaccuracy : 1.00
[5,    15] trainloss: 0.046, trainaccuracy : 1.00
[5,     1] testloss: 0.899, testaccuracy : 0.70
[5,     3] testloss: 0.879, testaccuracy : 0.66
[5,     5] testloss: 1.846, testaccuracy : 0.39
Finished Training
훈련부분






tensor([[14.,  4.,  7.],
        [ 6., 15.,  4.],
        [ 6.,  1.,  9.]])
Accuracy of Covid : 75 %
Accuracy of Normal : 100 %
Accuracy of Viral Pneumonia : 50 %
결과부분
