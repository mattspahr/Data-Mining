import Orange
import json


def associationRule():

    filename = 'jokeData.txt'
    raw_data = []

    with open(filename) as fil:
        for row in fil:
            raw_data = row

    # write data to the text file: data.basket
    f = open('data.basket', 'w')
    for item in raw_data:
        f.write(item + '\n')
    f.close()

    # Load data from the text file: data.basket
    data = Orange.data.Table("data.basket")

    # Identify association rules with supports at least 0.3
    rules = Orange.associate.AssociationRulesSparseInducer(data, support=.3)

    # print out rules
    print "%4s %4s  %s" % ("Supp", "Conf", "Rule")
    for r in rules[:]:
        print "%4.1f %4.1f  %s" % (r.support, r.confidence, r)


if __name__ == '__main__':
    associationRule()