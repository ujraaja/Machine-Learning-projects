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
#include <float.h>
using namespace std;

#define SD(i) scanf("%d", &(i))
#define SDD(i, j) scanf("%d%d", &(i), &(j))
#define SDDD(i, j, k) scanf("%d%d%d", &(i), &(j), &(k))
#define SDDDD(i, j, k, l) scanf("%d%d%d%d", &(i), &(j), &(k), &(l))
#define SL(i) scanf("%lld", &(i))
#define SC(i) scanf(" %c", &(i))
#define SS(i) scanf(" %s", &(i))
#define PD(i) printf("%d", i)
#define PL(i) printf("%lld", i)
#define PC(i) printf("%c", i)
#define PS(i) printf("%s", i)
#define PN printf("\n");

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

int numAttributes, targetIndex;
enum dataType {continuous, discrete};
vector<dataType> dataTypes;
vector<map<string, int> > discreteValuesMap;
vector<vector<double> > continuousValues;
vector<vector<string> > data10Fold[10], trainingData;
vector<bool> isFeatureSelected;
int K;
vector<pair<double, int> > topKIndices; //<distance, trainIndex>

bool isAcceptedChar(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '-';
}

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
    while(!isAcceptedChar(buf[0])) cin.getline(buf, 500);
    do {
        int index = 0;
        vector<string> dat;
        REP(i, numAttributes) {
            string t = "";
            while(!isAcceptedChar(buf[index])) index++;
            while(isAcceptedChar(buf[index])) t += buf[index], index++;
            dat.push_back(t);
           if(dataTypes[i] == continuous)
                continuousValues[i].push_back(atof(t.c_str()));
            else if(discreteValuesMap[i].find(t) == discreteValuesMap[i].end())
                discreteValuesMap[i][t] = discreteValuesMap[i].size()-1;
        }
        data10Fold[rand()%10].push_back(dat);
        cin.getline(buf, 500);
    } while(isAcceptedChar(buf[0]));
    REP(i, numAttributes)
        if(dataTypes[i] == continuous) {
            SRT(continuousValues[i]);
            continuousValues[i].resize(distance(continuousValues[i].begin(),unique(ALL(continuousValues[i]))));
        }
}

void normalizeData() {
    int index = 0;
    REP(i, numAttributes) {
        if(dataTypes[i] == continuous) {
            double mn = DBL_MAX, mx = -DBL_MAX;
            REP(j, continuousValues[i].size())
                mn = min(mn, continuousValues[i][j]), mx = max(mx, continuousValues[i][j]);
            REP(j, continuousValues[i].size())
                continuousValues[i][j] = (continuousValues[i][j] - mn) / (mx - mn);
            REP(k, 10) REP(j, data10Fold[k].size()) {
                double val = atof(data10Fold[k][j][i].c_str());
                val = (val - mn) / (mx - mn);
                data10Fold[k][j][i] = to_string(val);
            }
        }
    }
}

double distance(vector<string> p1, vector<string> p2) {
    double dist = 0;
    REP(i, numAttributes) {
        if(i == targetIndex || !isFeatureSelected[i]) continue;
        if(dataTypes[i] == continuous) dist += pow(atof(p1[i].c_str()) - atof(p2[i].c_str()), 2);
        else if(p1[i] != p2[i]) dist += 0.25;
    }
    return dist;
}

void consider(double dist, int index) {
    if(topKIndices.size() < K) topKIndices.push_back({dist, index});
    else {
        int mxInd = -1;
        double mxDist = -1;
        REP(i, K)
            if(mxDist < topKIndices[i].first)
                mxDist = topKIndices[i].first, mxInd = i;
        if(mxDist > dist)
            topKIndices[mxInd] = {dist, index};
    }
}

string classify(vector<string> data, int index) {
    topKIndices.clear();
    REP(j, index) {
        double dist = distance(trainingData[j], data);
        consider(dist, j);
    }
    map<string, double> classesCount;
    REP(j, topKIndices.size()) {
        string clas = trainingData[topKIndices[j].second][targetIndex];
        classesCount[clas] += 1/topKIndices[j].first;
    }
    double mxCount = -1;
    string mxClass;
    ECH(it, classesCount)
        if(mxCount < it->second)
            mxCount = it->second, mxClass = it->first;
    return mxClass;
}

void ntGrowth() {
    vector<vector<string> > temp = trainingData;
    int pos[trainingData.size()], neg[trainingData.size()], state[trainingData.size()];
    CLR(pos);
    CLR(neg);
    CLR(state); //0->initial, 1->acceptable, 2->discarded
    int consideredSamplesCount = 0;
    map<string, int> classFreq;
    REP(i, temp.size()) {
        pos[i] = 1;
        classFreq[temp[i][targetIndex]]++, consideredSamplesCount++;
        int mnIdx = -1;
        double mnDist = DBL_MAX;
        REP(j, i) {
            if(state[j] == 1) {
                double dist = distance(temp[j], temp[i]);
                if(mnDist > dist)
                    mnIdx = j, mnDist = dist;
            }
        }
        if(mnIdx != -1) {
            if(temp[mnIdx][targetIndex] == temp[i][targetIndex]) state[i] = 2;
            REP(j, i) {
                if(state[j] != 2 && distance(temp[j], temp[i]) <= mnDist) {
                    if(temp[j][targetIndex] == temp[i][targetIndex]) pos[j]++;
                    else neg[j]++;
                }
            }
        } else {
            int k = K;
            topKIndices.clear();
            K = rand()%(int)sqrt(trainingData.size()) + sqrt(trainingData.size());
            REP(j, i)
                if(state[j] != 2)
                    consider(distance(trainingData[i], trainingData[j]), j);
            for(auto it : topKIndices)
                if(temp[it.second][targetIndex] == temp[i][targetIndex]) pos[it.second]++;
                else neg[it.second]++;
            K = k;
        }
        REP(j, i) {
            if(state[j] != 2 && pos[j] + neg[j] != 0) {
                double ar = pos[j] / (double)(pos[j]+neg[j]);   //classificationAccuracy
                double aci = 1.96 * sqrt(ar * (1-ar) / (pos[j]+neg[j]));
                double cr = classFreq[temp[j][targetIndex]]/(double)consideredSamplesCount;  //classFrequency
                double cci = 1.96 * sqrt(cr * (1-cr) / consideredSamplesCount);
                if(ar - aci > cr + cci) state[j] = 1;
                else if(ar + aci < cr - cci) state[j] = 2;
                else state[j] = 0;
            }
        }
    }
    trainingData.clear();
    REP(i, temp.size())
        if(state[i] == 1)
            trainingData.push_back(temp[i]);
}

void featureElimination() { //Stepwise Backward Elimination
    int validationSetIndex = trainingData.size() * 2 / 3;
    int accr = 0;
    FOR(k, validationSetIndex, trainingData.size())
        if(trainingData[k][targetIndex] == classify(trainingData[k], validationSetIndex))
            accr++;
    while(true) {
        int accr1 = 0;
        int dropFeatureIndex = -1;
        RREP(i, numAttributes) {
            if(i == targetIndex || !isFeatureSelected[i]) continue;
            isFeatureSelected[i] = false;
            int accr2 = 0;
            FOR(k, validationSetIndex, trainingData.size())
                if(trainingData[k][targetIndex] == classify(trainingData[k], validationSetIndex))
                    accr2++;
            if(accr2 > accr1)
                accr1 = accr2, dropFeatureIndex = i;
            isFeatureSelected[i] = true;
        }
        if(accr < accr1)
            accr = accr1, isFeatureSelected[dropFeatureIndex] = false;
        else
            break;
    }
}

void featureSelection() { //Stepwise Forward Selection
    int validationSetIndex = trainingData.size() * 2 / 3;
    int accr = 0;
    REP(j, numAttributes) isFeatureSelected[j] = false;
    while(true) {
        int accr1 = 0;
        int selectFeatureIndex = -1;
        REP(i, numAttributes) {
            if(i == targetIndex || isFeatureSelected[i]) continue;
            isFeatureSelected[i] = true;
            int accr2 = 0;
            FOR(k, validationSetIndex, trainingData.size())
                if(trainingData[k][targetIndex] == classify(trainingData[k], validationSetIndex))
                    accr2++;
            if(accr2 > accr1)
                accr1 = accr2, selectFeatureIndex = i;
            isFeatureSelected[i] = false;
        }
        if(accr < accr1)
            accr = accr1, isFeatureSelected[selectFeatureIndex] = true;
        else
            break;
    }
}

void knn() {
    double totalAccuracy = 0;
    int numAttributesSelected = 0;
    int numSamples = 0, numSamplesAfterNT = 0;
    vector<double> accuracies;
    isFeatureSelected.resize(numAttributes);
    REP(i, 10) {
        REP(j, numAttributes) isFeatureSelected[j] = true;
        int goodClassificationCount = 0;
        trainingData.clear();
        REP(j, 10) if(i != j) trainingData.insert(trainingData.end(), data10Fold[j].begin(), data10Fold[j].end());
        //numSamples += trainingData.size();
        //ntGrowth();
        //numSamplesAfterNT += trainingData.size();
        //featureSelection();
        //featureElimination();
        REP(k, data10Fold[i].size())
            if(data10Fold[i][k][targetIndex] == classify(data10Fold[i][k], trainingData.size()))
                goodClassificationCount++;
        totalAccuracy += (double) goodClassificationCount / data10Fold[i].size();
        accuracies.push_back((double) goodClassificationCount / data10Fold[i].size());
        //REP(j, numAttributes) if(isFeatureSelected[j] && j != targetIndex) numAttributesSelected++;
    }
    {
        double meanAccuracy = totalAccuracy/10;
        double sum = 0;
        for(auto t : accuracies) sum += pow(meanAccuracy-t, 2);
        sum /= 10;
        sum = sqrt(sum);
        cout<<"std dev: "<<sum<<endl;
    }
    //cout<<"Number of attributes: "<<numAttributes-1<<endl;
    //cout<<"Number of selected attributes: "<<numAttributesSelected/10.0<<endl;
    //cout<<"Number of samples before NT: "<<numSamples/10.0<<endl;
    //cout<<"Number of samples after NT: "<<numSamplesAfterNT/10.0<<endl;
    cout<<"Average accuracy: "<<totalAccuracy * 10<<endl;
}

void createTest() {
    REP(i, 1000) {
        double x = rand() / (double)RAND_MAX;
        double y = rand() / (double)RAND_MAX;
        double prob = rand() / (double)RAND_MAX;
        cout<<x<<","<<y<<","<<(prob >= 0.01?(x<y?0:1):(x>y?0:1))<<endl;
    }
}

int main() {
    K = 7;
//    freopen("car.data", "r", stdin);
//    freopen("iris.data", "r", stdin);
//    freopen("king-rook-vs-king-pawn.data", "r", stdin);
//    freopen("thoracic-surgery.data", "r", stdin);
//    freopen("tic-tac-toe.data", "r", stdin);
    freopen("wine.data", "r", stdin);
    readFile();
    normalizeData();
    knn();
}
