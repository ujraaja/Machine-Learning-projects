#include <vector>
#include <list>
#include <map>
#include <set>
#include <deque>
#include <stack>
#include <bitset>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <queue>
#include <cassert>
using namespace std;

#define FOR(i,a,b) for(int i=a;i<b;++i)
#define SFOR(i,a,b,c) for(int i=a;i<b;i+=c)
#define REP(i,n) FOR(i,0,n)
#define RFOR(i,a,b) for(int i=a;i>=b;i--)
#define RREP(i,n) RFOR(i,n-1,0)
#define ECH(it, v) for(auto it=v.begin();it!=v.end();++it)
#define ALL(x) (x).begin(),(x).end()
#define SRT(x) sort(ALL(x))
#define CLR(x) memset(x,0,sizeof(x))
#define SET(x) memset(x,-1,sizeof(x))
#define MOD 1000000007
typedef unsigned int UI;
typedef long long LL;
typedef unsigned long long UL;
typedef vector<int> VI;
typedef vector<VI> VVI;
typedef vector<string> VS;
typedef pair<int, int> PI;

struct node {
    int splitAttrIndex;
    string classValue;
    vector<string> values;
    vector<node*> nodes;
    vector<int> classesCount;
};

node *root;
int numAttributes, targetIndex;
double tempSplitValue;
enum dataType {continuous, discrete};
const string missingValue = "?";
vector<dataType> dataTypes;
vector<map<string, int>> discreteValuesMap;
vector<vector<double>> continuousValues;
map<string, int> classes;
vector<vector<string>> data, data10Fold[10], validationSet; //data -> training data

void readFile() {
    cin>>numAttributes>>targetIndex;
    dataTypes.resize(numAttributes);
    discreteValuesMap.resize(numAttributes);
    continuousValues.resize(numAttributes);
    REP(i, numAttributes) {
        char c;
        cin>>c;
        if(c == 'c') dataTypes[i] = continuous;
        else dataTypes[i] = discrete;
    }
    char buf[500];
    cin.getline(buf, 500);
    while(buf[0] == 0) cin.getline(buf, 500);
    do {
        int index = 0;
        vector<string> dat;
        REP(i, numAttributes) {
            string t = "";
            if(buf[index] == ',') index++;
            while(buf[index] == ' ') index++;
            while(buf[index] != ',' && buf[index] != 0) t += buf[index], index++;
            dat.push_back(t);
            if(t == missingValue) continue;
            if(dataTypes[i] == continuous)
                continuousValues[i].push_back(atof(t.c_str()));
            else if(discreteValuesMap[i].find(t) == discreteValuesMap[i].end())
                discreteValuesMap[i][t] = discreteValuesMap[i].size()-1;
        }
        int isValidation = rand() % 3;
        if(isValidation == 0) validationSet.push_back(dat);
        else data10Fold[rand()%10].push_back(dat);
        cin.getline(buf, 500);
    } while(buf[0]);
    REP(i, numAttributes)
        if(dataTypes[i] == continuous) {
            SRT(continuousValues[i]);
            auto it = unique(ALL(continuousValues[i]));
            continuousValues[i].resize(distance(continuousValues[i].begin(),it));
        }
}

void determineMissingValues() {
    REP(i, 10) {
        for(auto d : data10Fold[i]) {
            REP(j, numAttributes) {
                if(d[j] == missingValue) {
                    map<string, int> mp;
                    REP(ii, 10)
                        for(auto dd : data10Fold[ii])
                            if(dd[targetIndex] == d[targetIndex])
                                mp[dd[j]]++;
                    int mx = -1;
                    for(auto m : mp)
                        if(mx < m.second)
                            mx = m.second, d[j] = m.first;
                }
            }
        }
    }
}

/*
Max(infoGain) => Max(-sum(proportion * entropy)) => Min(sum(proportion * entropy)) => Max(sum(proportion * sum(proportion * log2)))
This function returns (sum(proportion * sum(proportion * log2)))
*/
double entropy(vector<int> samplesIndices, int attrIndex) {
    if(dataTypes[attrIndex] == continuous) {
        vector<double> attrValues;
        for(int i : samplesIndices) attrValues.push_back(atof(data[i][attrIndex].c_str()));
        SRT(attrValues);
        auto it = unique(ALL(attrValues));
        attrValues.resize(distance(attrValues.begin(),it));
        if(attrValues.size() == 1) {
            tempSplitValue = attrValues[0];
            return 0;
        }
        double maxEntropy = -MOD;
        for(int i = 0; i < attrValues.size()-1; i++) {
            double splitValue = (attrValues[i] + attrValues[i+1]) / 2;
            int lessCount = 0, moreCount = 0;
            for(int j : samplesIndices)
                if(atof(data[j][attrIndex].c_str()) < splitValue) lessCount++;
                else moreCount++;
            double entropy = lessCount * log2(lessCount/(double)(lessCount+moreCount))
                        + moreCount * log2(moreCount/(double)(lessCount+moreCount));
            if(maxEntropy < entropy)
                maxEntropy = entropy, tempSplitValue = splitValue;
        }
        return maxEntropy;
    } else {
        set<string> attrValues;
        double entropy = 0;
        for(int i : samplesIndices) attrValues.insert(data[i][attrIndex]);
        for(string value : attrValues) {
            map<string, int> classFreq;
            int cnt = 0;
            for(int i : samplesIndices)
                if(value == data[i][attrIndex]) {
                    classFreq[data[i][targetIndex]]++;
                    cnt++;
                }
            for(auto entry : classFreq)
                entropy +=  log2(entry.second/(double) cnt) * entry.second;
        }
        return entropy;
    }
}

void ID3(node* root, vector<int> samplesIndices, vector<int> attrIndices) {
    bool flag = true;
    for(int i : samplesIndices)
        if(data[samplesIndices[0]][targetIndex] != data[i][targetIndex])
            flag = false;
    if(flag) {      //all samples belong to the same class
        root->splitAttrIndex = -1;
        root->classValue = data[samplesIndices[0]][targetIndex];
        return;
    }
    {   //check if only one value for an attribute, if so, it can't be split
        REP(i, attrIndices.size()) {
            flag = true;
            for(int j : samplesIndices)
                if(data[samplesIndices[0]][attrIndices[i]] != data[j][attrIndices[i]])
                    flag = false;
            if(flag) {
                attrIndices.erase(attrIndices.begin() + i);
                i--;
            }
        }
    }
    if(attrIndices.size() == 0) {   //no more attributes
        map<string, int> mp;
        for(int i : samplesIndices)
            mp[data[i][targetIndex]]++;
        int mxFreq = -1;
        string mxVal;
        for(auto it : mp)
            if(it.second > mxFreq) mxVal = it.first, mxFreq = it.second;
        root->splitAttrIndex = -1;
        root->classValue = mxVal;
        return;
    }
    double bestIG = -MOD, bestSplitValue;
    int bestAttrID = -1;
    for(int i : attrIndices) {
        double ig = entropy(samplesIndices, i);
        if(ig > bestIG) bestIG = ig, bestAttrID = i, bestSplitValue = tempSplitValue;
    }
    root->splitAttrIndex = bestAttrID;
    if(dataTypes[bestAttrID] == continuous) {
        root->values.push_back(to_string(bestSplitValue));
        vector<int> sampleIDs1, sampleIDs2;
        for(int i : samplesIndices)
            if(atof(data[i][bestAttrID].c_str()) < bestSplitValue) sampleIDs1.push_back(i);
            else sampleIDs2.push_back(i);
        node* t = new node;
        root->nodes.push_back(t);
        ID3(t, sampleIDs1, attrIndices);
        t = new node;
        root->nodes.push_back(t);
        ID3(t, sampleIDs2, attrIndices);
    } else {
        map<string, int> mp;
        for(int i : samplesIndices)
            mp[data[i][targetIndex]]++;
        int mx = -1;
        for(auto t : mp)
            if(mx < t.second)
                root->classValue = t.first, mx = t.second;
        set<string> attrValues;
        for(int i : samplesIndices) attrValues.insert(data[i][bestAttrID]);
        REP(i, attrIndices.size())
            if(attrIndices[i] == bestAttrID) {
                attrIndices.erase(attrIndices.begin() + i);
                i = MOD;
            }
        for(string value : attrValues) {
            node* t = new node;
            root->nodes.push_back(t);
            root->values.push_back(value);
            vector<int> sampleIDs;
            for(int i : samplesIndices)
                if(data[i][bestAttrID] == value)
                    sampleIDs.push_back(i);
            ID3(t, sampleIDs, attrIndices);
        }
    }
}

string classify(node* root, vector<string> data) {
    if(root->splitAttrIndex == -1) return root->classValue;
    if(dataTypes[root->splitAttrIndex] == continuous) {
        if(data[root->splitAttrIndex] < root->values[0])
            return classify(root->nodes[0], data);
        else
            return classify(root->nodes[1], data);
    } else {
        REP(i, root->values.size())
            if(data[root->splitAttrIndex] == root->values[i])
                return classify(root->nodes[i], data);
        int majorityClass = -1, mx = -1;
        REP(i, classes.size())
            if(mx < root->classesCount[i])
                mx = root->classesCount[i], majorityClass = i;
        for(auto t : classes)
            if(t.second == majorityClass)
                return t.first;
    }
}

void destroyTree(node* root) {
    if(root == NULL) return;
    for(node* n : root->nodes)
        destroyTree(n);
    free(root);
}

double pruneTree(node* root) {  //Reduced error pruning
    int numSamples = 0, majorityClassSamples = -1, majorityClass;
    REP(i, classes.size()) {
        numSamples += root->classesCount[i];
        if(majorityClassSamples < root->classesCount[i])
            majorityClassSamples = root->classesCount[i], majorityClass = i;
    }
    double prunedError, notPrunedError = 0;
    prunedError = (numSamples - majorityClassSamples)/(double) numSamples;
    for(node* t : root->nodes)
        notPrunedError += pruneTree(t)/numSamples;
    int c = 0;
    REP(i, classes.size())
        if(i != majorityClass)
            for(node* t : root->nodes) c += t->classesCount[i];
    c = numSamples - c - majorityClassSamples;
    notPrunedError += c/(double) numSamples;
    if(notPrunedError >= prunedError) {
        //if(root->splitAttrIndex != -1) cout<<notPrunedError<<":"<<prunedError<<endl;
        for(node* t : root->nodes) destroyTree(t);
        root->splitAttrIndex = -1;
        root->nodes.clear();
        root->values.clear();
        for(auto t : classes)
            if(t.second == majorityClass) root->classValue = t.first;
        notPrunedError = prunedError;
    }
    return notPrunedError * numSamples;
}

//double pruneTree(node* root) {  //Minimum error pruning
//    int numSamples = 0, majorityClassSamples = -1, majorityClass;
//    REP(i, classes.size()) {
//        numSamples += root->classesCount[i];
//        if(majorityClassSamples < root->classesCount[i])
//            majorityClassSamples = root->classesCount[i], majorityClass = i;
//    }
//    double prunedError, notPrunedError = 0;
//    prunedError = (numSamples - majorityClassSamples + classes.size() - 1)/(double)(numSamples + classes.size());
//    for(node* t : root->nodes)
//        notPrunedError += pruneTree(t)/numSamples;
//    int c = 0;
//    REP(i, classes.size())
//        if(i != majorityClass)
//            for(node* t : root->nodes) c += t->classesCount[i];
//    c = numSamples - c - majorityClassSamples;
//    notPrunedError += (c)/(double)(numSamples);
//    if(notPrunedError >= prunedError) {
//        if(root->splitAttrIndex != -1) cout<<notPrunedError<<":"<<prunedError<<endl;
//        for(node* t : root->nodes) destroyTree(t);
//        root->splitAttrIndex = -1;
//        root->nodes.clear();
//        root->values.clear();
//        for(auto t : classes)
//            if(t.second == majorityClass) root->classValue = t.first;
//        notPrunedError = prunedError;
//    }
//    return notPrunedError * numSamples;
//}

/*
Uses recursive bottom-up approach, to simplify implementation, however the principle of the approach is maintained;
at each node, we check if the node can be pruned, if so we prune it and return the error for non-pruned case to the parent.
If the parent is pruned, the values in the children doesn't matter; if the parent is not pruned, whichever children can be
pruned had already been pruned.
*/
//double pruneTree(node* root) {  //Pessimistic pruning
//    int numSamples = 0, majorityClassSamples = -1, majorityClass;
//    REP(i, classes.size()) {
//        numSamples += root->classesCount[i];
//        if(majorityClassSamples < root->classesCount[i]) majorityClassSamples = root->classesCount[i], majorityClass = i;
//    }
//    double prunedError = numSamples - majorityClassSamples + 0.5, notPrunedError = 0;
//    if(root->splitAttrIndex == -1) return prunedError;  //leaf node
//    for(node* t : root->nodes) notPrunedError += pruneTree(t);
//    if(numSamples == 0) return notPrunedError;
//    int c = 0;
//    REP(i, classes.size())
//        if(i != majorityClass)
//            for(node* t : root->nodes) c += t->classesCount[i];
//    c = numSamples - c - majorityClassSamples;
//    notPrunedError += c;
//    double stdError = sqrt(notPrunedError * max(0.0,(numSamples-notPrunedError)) / (double)numSamples);
//    if(notPrunedError > prunedError - stdError / numSamples) {
//        for(node* t : root->nodes) destroyTree(t);
//        root->splitAttrIndex = -1;
//        root->nodes.clear();
//        root->values.clear();
//        for(auto t : classes)
//            if(t.second == majorityClass) root->classValue = t.first;
//    }
//    return notPrunedError;
//}

void fillTreeWithValidationData(vector<int> sampleIndices, node* root) {
    root->classesCount.clear();
    root->classesCount.resize(classes.size(), 0);
    if(root->splitAttrIndex == -1) {
        for(int index : sampleIndices)
            root->classesCount[classes[validationSet[index][targetIndex]]]++;
        return;
    }
    if(dataTypes[root->splitAttrIndex] == discrete) {
        REP(i, root->nodes.size()) {
            vector<int> indices;
            REP(j, sampleIndices.size())
                if(validationSet[sampleIndices[j]][root->splitAttrIndex] == root->values[i]) {
                    indices.push_back(sampleIndices[j]);
                    sampleIndices.erase(sampleIndices.begin() + j);
                    j--;
                }
            fillTreeWithValidationData(indices, root->nodes[i]);
        }
        for(int index : sampleIndices)
            root->classesCount[classes[validationSet[index][targetIndex]]]++;
    } else {
        vector<int> indices1, indices2;
        double splitValue = atof(root->values[0].c_str());
        for(int index : sampleIndices)
            if(atof(validationSet[index][root->splitAttrIndex].c_str()) < splitValue) indices1.push_back(index);
            else indices2.push_back(index);
        fillTreeWithValidationData(indices1, root->nodes[0]);
        fillTreeWithValidationData(indices2, root->nodes[1]);
    }
    for(node* t : root->nodes)
        REP(i, root->classesCount.size())
            root->classesCount[i] += t->classesCount[i];
}

int numNodes;
void countNodes(node* root) {
    numNodes++;
    for(node* t : root->nodes)
        countNodes(t);
}

void chk(node* root) {
    if(root->splitAttrIndex == -1) {
        assert(root->nodes.size() == 0);
        assert(root->values.size() == 0);
    } else {
        REP(i, classes.size()) {
            int c = 0;
            for(node * t : root->nodes) c += t->classesCount[i];
            assert(root->classesCount[i] >= c);
        }
    }
    assert(root->classesCount.size() == classes.size());
    for(node * t : root->nodes) chk(t);
}

double majorityClassifierAccuracy(int data10FoldIndex) {
    vector<int> classFrequency(classes.size(), 0);
    REP(i, data10Fold[data10FoldIndex].size())
        classFrequency[classes[data10Fold[data10FoldIndex][i][targetIndex]]]++;
    return *max_element(ALL(classFrequency))/(double) accumulate(ALL(classFrequency), 0) * 100;
}

void testID3() {
    double totalAccuracty = 0, prunedAccuracy = 0, majorityAccuracy = 0;
    int treeSize1 = 0, treeSize2 = 0;
    vector<int> attrIndices;
    REP(i, numAttributes) if(i != targetIndex) attrIndices.push_back(i);
    REP(i, 10) REP(j, data10Fold[i].size())
        if(classes.find(data10Fold[i][j][targetIndex]) == classes.end())
            classes[data10Fold[i][j][targetIndex]] = classes.size()-1;
    REP(j, validationSet.size())
        if(classes.find(validationSet[j][targetIndex]) == classes.end())
            classes[validationSet[j][targetIndex]] = classes.size()-1;
    REP(j, 10) {
        vector<int> samplesIndices;
        root = new node;
        data.clear();
        REP(i, 10) if(i != j) data.insert(data.end(), data10Fold[i].begin(), data10Fold[i].end());
        REP(i, data.size()) samplesIndices.push_back(i);
        ID3(root, samplesIndices, attrIndices);

        samplesIndices.clear();
//validationSet = data;       //to be used only for pessimistic pruning
        REP(i, validationSet.size()) samplesIndices.push_back(i);
        fillTreeWithValidationData(samplesIndices, root);

        int correct = 0, wrong = 0;
        for(auto dat : data10Fold[j])
            if(dat[targetIndex] != classify(root, dat)) wrong++;
            else correct++;
        totalAccuracty += correct*100.0/(correct+wrong);

        numNodes = 0;
        countNodes(root);
        treeSize1 += numNodes;
        pruneTree(root);

        numNodes = 0;
        countNodes(root);
        treeSize2 += numNodes;

        correct = 0, wrong = 0;
        for(auto dat : data10Fold[j])
            if(dat[targetIndex] != classify(root, dat)) wrong++;
            else correct++;
        prunedAccuracy += correct*100.0/(correct+wrong);

        majorityAccuracy += majorityClassifierAccuracy(j);
        destroyTree(root);
    }
    cout<<"Accuracy before pruning: "<<totalAccuracty/10<<endl;
    cout<<"Pruned accuracy: "<<prunedAccuracy/10<<endl;
    cout<<"Majority accuracy: "<<majorityAccuracy/10<<endl;
    cout<<"Tree size before pruning: "<<treeSize1/10.0<<endl;
    cout<<"Tree size after pruning: "<<treeSize2/10.0<<endl;
    int numTestSamples = 0;
    REP(i, 10) numTestSamples += data10Fold[i].size();
    prunedAccuracy /= 10 * 100;
    double confInterval = sqrt(prunedAccuracy*(1-prunedAccuracy)/(numTestSamples/10.0));
    cout<<"Confidence Interval: "<<prunedAccuracy-confInterval<<" to "<<prunedAccuracy+confInterval<<endl;
}

void printTree(node* root, int level) {
    if(root == NULL) return;
    for(int i = 0; i < root->values.size(); i++) {
        REP(j, level) cout<<"  ";
        cout<<root->splitAttrIndex<<" = "<<root->values[i]<<endl;
        printTree(root->nodes[i], level+1);
    }
    if(root->splitAttrIndex == -1)
        cout<<"\t\t-->"<<root->classValue<<endl;
}

int main() {
    freopen("car.data", "r", stdin);
//    freopen("iris.data", "r", stdin);
//    freopen("king-rook-vs-king-pawn.data", "r", stdin);
//    freopen("tic-tac-toe.data", "r", stdin);
//    freopen("wine.data", "r", stdin);
    readFile();
    determineMissingValues();
    testID3();
    //printTree(root, 0);
}
