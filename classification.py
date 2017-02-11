from sklearn import tree

    with open('house-votes-84.data', 'r') as inputFile:
        lines = inputFile.readlines()

    X = []
    Y = []
    X_unseen = []
    Y_unseen = []

    count = 0
    for line in lines:
        if not '?' in line:
            # print(line)
            count = count + 1
            values = line.split(',')
            party = values[0]

            if 'democrat' in party:
                klass = 1
            if 'republican' in party:
                klass = 0

            instanceValues = []
            for index in enumerate(values, start=1):
                if index[0] != 1:
                    if 'n' in index[1]:
                        instanceValues.append(0)
                    if 'y' in index[1]:
                        instanceValues.append(1)

            # Take every other sample and set aside for validation of our decision tree
            # In other words, don't include these rows in our training data
            if count % 2 == 0:
                Y_unseen.append(klass)
                X_unseen.append(instanceValues)
            else:
                Y.append(klass)
                X.append(instanceValues)

    print ("Total instances:", count)
    print ("Training instances:", len(X))
    print ("Validation instances:", len(X_unseen))

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    predictions = clf.predict(X_unseen)

    # print ("Predictions:", predictions)
    # print ("Actual vals:", Y_unseen)

    wrongPredictions = 0
    for index, val in enumerate(predictions):
        if Y_unseen[index] != val:
            wrongPredictions = wrongPredictions + 1
    print ("Correctly predicted %s out of %s instances" % ((len(Y_unseen) - wrongPredictions), len(Y_unseen)))
