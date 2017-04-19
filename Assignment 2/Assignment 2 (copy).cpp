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
int numHiddenLayers, numHiddenNodes[2], minEpochs;
double eeta, alpha; //alpha -> momentum rate
const string missingValue = "?";
enum dataType {continuous, discrete};
vector<dataType> dataTypes;
vector<map<string, int> > discreteValuesMap;
vector<vector<double> > continuousValues;
map<string, int> classes;
vector<vector<string> > data10Fold[10];
vector<vector<double> > intData10Fold[10], class10Fold[10], x, y, weights[3], delta[3]; //x->input, y->output

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
            if(t == missingValue) continue;
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

void convertData() {
    REP(i, 10) {
        REP(j, data10Fold[i].size()) {
            vector<double> data, output;
            REP(k, numAttributes) {
                if(k == targetIndex) {
                    if(dataTypes[k] == continuous) output.push_back(atof(data10Fold[i][j][k].c_str()));
                    else {
                        REP(l, discreteValuesMap[k][data10Fold[i][j][k]]) output.push_back(0);
                        output.push_back(1);
                        FOR(l, discreteValuesMap[k][data10Fold[i][j][k]]+1, discreteValuesMap[k].size()) output.push_back(0);
                    }
                } else {
                    if(dataTypes[k] == continuous) data.push_back(atof(data10Fold[i][j][k].c_str()));
                    else {
//                        int n = discreteValuesMap[k].size();
//                        int totalCount = log2(2*n-1), value = discreteValuesMap[k][data10Fold[i][j][k]], index = 0;
//                        while(value) {
//                            data.push_back(value&1);
//                            value >>= 1;
//                            index++;
//                        }
//                        while(index < totalCount) {
//                            data.push_back(0);
//                            index++;
//                        }
                        REP(l, discreteValuesMap[k][data10Fold[i][j][k]]) data.push_back(0);
                        data.push_back(1);
                        FOR(l, discreteValuesMap[k][data10Fold[i][j][k]]+1, discreteValuesMap[k].size()) data.push_back(0);
                    }
                }
            }
            intData10Fold[i].push_back(data);
            class10Fold[i].push_back(output);
        }
    }
}

void normalizeData() {
    int index = 0;
    REP(i, numAttributes) {
        if(i == targetIndex) continue;
        if(dataTypes[i] == discrete) {
            index += continuousValues[i].size();
            continue;
        }
        double mn = DBL_MAX, mx = -DBL_MAX;
        REP(k, 10) REP(j, intData10Fold[k].size())
            mn = min(mn, intData10Fold[k][j][index]), mx = max(mx, intData10Fold[k][j][index]);
        REP(k, 10) REP(j, intData10Fold[k].size())
            intData10Fold[k][j][index] = -1 + 2*(intData10Fold[k][j][index]-mn)/(mx-mn);
        index++;
    }
    if(dataTypes[targetIndex] == continuous) {
        double mn = DBL_MAX, mx = -DBL_MAX;
        REP(k, 10) REP(j, class10Fold[k].size())
            mn = min(mn, class10Fold[k][j][0]), mx = max(mx, class10Fold[k][j][0]);
        REP(k, 10) REP(j, class10Fold[k].size())
            class10Fold[k][j][0] = -1 + 2*(class10Fold[k][j][0]-mn)/(mx-mn);
    }
}

vector<vector<double> > transpose(vector<vector<double> > v){
    vector<vector<double> > ret;
    if(v.size() == 0 || v[0].size() == 0) return ret;
    REP(i, v[0].size()) {
        vector<double> t;
        REP(j, v.size()) t.push_back(v[j][i]);
        ret.push_back(t);
    }
    return ret;
}

vector<vector<double> > matrixMultiply(vector<vector<double> > a, vector<vector<double> > b){
    vector<vector<double> > ret(a.size(), vector<double>(b[0].size(), 0));
    REP(i, ret.size())
        REP(j, ret[0].size())
            REP(k, b.size())
                ret[i][j] += a[i][k] * b[k][j];
    return ret;
}

vector<vector<double> > elementWiseMultiply(vector<vector<double> > a, vector<vector<double> > b){
    REP(i, a.size()) REP(j, a[0].size()) a[i][j] *= b[i][j];
    return a;
}

vector<vector<double> > matrixSub(vector<vector<double> > a, vector<vector<double> > b){
    vector<vector<double> > ret(a.size(), vector<double>(b[0].size()));
    REP(i, ret.size())
        REP(j, ret[0].size())
            ret[i][j] = a[i][j] - b[i][j];
    return ret;
}

vector<vector<double> > ones(int x, int y) {
    return vector<vector<double> >(x, vector<double>(y, 1));
}

double epoch() {
    double err = 0;
    vector<vector<double> > hiddenAct[numHiddenLayers+2], del[numHiddenLayers+1];
    hiddenAct[0] = x;
    REP(k, numHiddenLayers+1) {
        hiddenAct[k+1] = matrixMultiply(hiddenAct[k], transpose(weights[k]));
        REP(j, hiddenAct[k+1].size()) REP(l, hiddenAct[k+1][j].size())
            hiddenAct[k+1][j][l] = 1/(1+exp(-hiddenAct[k+1][j][l]));
        if(k+1 < numHiddenLayers+1)
            REP(j, hiddenAct[k+1].size()) hiddenAct[k+1][j].push_back(1); //adding bias
    }
    del[numHiddenLayers] = matrixSub(y, hiddenAct[numHiddenLayers+1]);
    REP(i, del[numHiddenLayers].size())
        REP(j, del[numHiddenLayers][0].size())
            err += del[numHiddenLayers][i][j] * del[numHiddenLayers][i][j];
    del[numHiddenLayers] = elementWiseMultiply(del[numHiddenLayers], hiddenAct[numHiddenLayers+1]);
    del[numHiddenLayers] = elementWiseMultiply(del[numHiddenLayers], matrixSub(ones(del[numHiddenLayers].size(), del[numHiddenLayers][0].size()),hiddenAct[numHiddenLayers+1]));
    REP(j, del[numHiddenLayers][0].size())  //adding momentum
        REP(k, hiddenAct[numHiddenLayers][0].size())
            delta[numHiddenLayers][j][k] *= alpha;
    REP(i, x.size())
        REP(j, del[numHiddenLayers][0].size())
            REP(k, hiddenAct[numHiddenLayers][0].size())
                delta[numHiddenLayers][j][k] += eeta * hiddenAct[numHiddenLayers][i][k] * del[numHiddenLayers][i][j];
    REP(j, hiddenAct[numHiddenLayers+1][0].size())
        REP(k, hiddenAct[numHiddenLayers][0].size())
            weights[numHiddenLayers][j][k] += delta[numHiddenLayers][j][k];
    RREP(i, numHiddenLayers) {
        del[i] = matrixMultiply(del[i+1], weights[i+1]);
        del[i] = elementWiseMultiply(del[i], hiddenAct[i+1]);
        del[i] = elementWiseMultiply(del[i], matrixSub(ones(hiddenAct[i+1].size(), hiddenAct[i+1][0].size()),hiddenAct[i+1]));
        REP(j, del[i][0].size()-1)  //adding momentum
            REP(k, hiddenAct[i][0].size())
                delta[i][j][k] *= alpha;
        REP(l, x.size())
            REP(j, del[i][0].size()-1)  //ignore input weights to bias
                REP(k, hiddenAct[i][0].size())
                    delta[i][j][k] += eeta * hiddenAct[i][l][k] * del[i][l][j];
        REP(j, del[i][0].size()-1)  //ignore input weights to bias
            REP(k, hiddenAct[i][0].size())
                weights[i][j][k] += delta[i][j][k];
    }
    return err/y.size()/y[0].size();
}

void trainNeural() {
    assert(x.size() && y.size());
    if(numHiddenLayers == 2) {
        weights[0].resize(numHiddenNodes[0], vector<double>(x[0].size()+1));
        weights[1].resize(numHiddenNodes[1], vector<double>(numHiddenNodes[0]+1));
        weights[2].resize(y[0].size(), vector<double>(numHiddenNodes[1]+1));
    } else if(numHiddenLayers == 1) {
        weights[0].resize(numHiddenNodes[0], vector<double>(x[0].size()+1));
        weights[1].resize(y[0].size(), vector<double>(numHiddenNodes[0]+1));
    } else {
        weights[0].resize(y[0].size(), vector<double>(x[0].size()+1));
    }
    REP(i, numHiddenLayers+1)
        REP(j, weights[i].size())
            REP(k, weights[i][0].size())
                weights[i][j][k] = rand()/(double)RAND_MAX;
    REP(i, x.size()) x[i].push_back(1); //adding bias
    REP(k, numHiddenLayers+1)
        delta[k].resize(weights[k].size(), vector<double>(weights[k][0].size()));
    int numEpochs = 0, minErrEpoch = minEpochs;
    double minErr = -DBL_MAX;
    for(; numEpochs < minEpochs; numEpochs++) {
        double err = epoch();
        cout<<"MSE: "<<err<<endl;
        minErr = min(minErr, err);
    }
    while(numEpochs < 1000000) {
        double err = epoch();
        cout<<"MSE: "<<err<<endl;
        if(err < minErr) {
            minErr = err;
            minErrEpoch = numEpochs;
        } else if(numEpochs >= 2*minErrEpoch)
            break;
        numEpochs++;
    }
}

int correctClassification = 0;

void classify() {
    vector<vector<double> > curAct = x;
    REP(i, numHiddenLayers+1) {
        REP(j, curAct.size()) curAct[j].push_back(1);   //adding bias
        curAct = matrixMultiply(curAct, transpose(weights[i]));
        if(i != numHiddenLayers)
            REP(j, curAct.size())
                REP(k, curAct[j].size())
                    curAct[j][k] = 1/(1+exp(-curAct[j][k]));
    }
    REP(i, curAct.size()) {
        double mx = -DBL_MAX, clas = -1;
        REP(j, curAct[i].size())
            if(curAct[i][j] > mx)
                mx = curAct[i][j], clas = j;
        if(y[i][clas] == 1) correctClassification++;
    }
}

void neural() {
    double totalAccuracy = 0;
    REP(i, 10) {
        x.clear();
        y.clear();
        REP(j, 10)
            if(i != j) {
                x.insert(x.end(), intData10Fold[i].begin(), intData10Fold[i].end());
                y.insert(y.end(), class10Fold[i].begin(), class10Fold[i].end());
            }
        trainNeural();

        correctClassification = 0;
        x = intData10Fold[i];
        y = class10Fold[i];
        classify();
        totalAccuracy += correctClassification*100.0/y.size();
    }
    cout<<"Average accuracy: "<<totalAccuracy/10<<endl;
}

int main() {
    freopen("car.data", "r", stdin);
//    freopen("iris.data", "r", stdin);
//    freopen("king-rook-vs-king-pawn.data", "r", stdin);
//    freopen("tic-tac-toe.data", "r", stdin);
//    freopen("wine.data", "r", stdin);
//    freopen("output.txt", "w", stdout);
    readFile();
    convertData();
    normalizeData();
    numHiddenLayers = 0;
    //numHiddenNodes[0] = log2(intData10Fold[0][0].size()*2-1);
    numHiddenNodes[0] = 5;
    numHiddenNodes[1] = 5;
    minEpochs = 500;
    eeta = 0.03;
    alpha = 0.1;
    neural();
}
