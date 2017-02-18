#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <string>
#include <list>
#include <stdio.h>
#include <math.h>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

using std::vector;
using std::string;
using std::list;

static const bool DEBUG_PRINT = true;
static const float PT_DIST = 2.0; //Maximum allowed distance between points of object

vector<int> foo;
vector<vector<vector<float> > > segObjects;



struct depthCmp
{
    bool operator()(const vector<float>& lhs,
	   const vector<float>& rhs) const
    {
	   return lhs[0] < rhs[0];
    }
};

void sortDepth(vector<vector<float> > toSort) {
    //sort notSeg by depth
    std::sort(toSort.begin(), toSort.end(), depthCmp());
}

//Calculates distance between 2 points
float distance(vector<float> v1, vector<float> v2) { 
    float sum = 0;
    for (int i = 1; i < 4; i++) {
	   sum += (float) abs(pow(v1.at(i) - v2.at(i),3.0));
    }
    return sqrt(sum);
}

void writeCloud(vector<vector<float> > cloud, string name) {
    std::ofstream outputFile(name);
    std::ostream_iterator<float> output_iterator(outputFile, " ");
    BOOST_FOREACH(vector<float> point, cloud){
	   std::copy(point.begin(), point.end(), output_iterator);
	   outputFile << "\n";
    }
    outputFile.close();
}

void writeCloudList(vector<vector<vector<float> > > cloudList ){
    int cloudNum = 0;
    BOOST_FOREACH(vector<vector<float> > toWrite, cloudList) {
	   writeCloud(toWrite, "./processed/proccloud"+std::to_string(cloudNum)+".txt");
	   cloudNum++;
    }

}

vector<vector<float> > readCloud(string fileName) {
    vector<vector<float> > cloud;
    int rowNum = 0;
    std::ifstream inputFile(fileName);
    while (true) {
	   
		  vector<float> tmpVec;
		  float tmpFloat;

			 for (int j = 0; j < 4; j++) {
				inputFile >> tmpFloat;
				if (DEBUG_PRINT) printf("data: %f\n", tmpFloat);
				if (inputFile.eof()) return cloud;
				tmpVec.push_back(tmpFloat);
			 }
			 cloud.push_back(tmpVec);
			 if (DEBUG_PRINT) printf("Row Number: %d\n", ++rowNum);
    }

    
}
vector<vector<vector<float> > >  segmentCloudEfficient(list<vector<float> > toSegment) {
    list<vector<float> > notSeg = toSegment;
    list<vector<float> >::iterator nsIt;

    vector<vector<vector<float> > > objList;

    //segment object
    

    notSeg.sort(depthCmp());
    
    while (!notSeg.empty()) {
	   vector<vector<float> >inCloud;
	   vector<float> closestPoint = notSeg.front();
	   inCloud.push_back(closestPoint);
	   notSeg.pop_front();

	   for (int i = 0; i < (int)inCloud.size(); i++) {//not at end of incloud

		  vector<float> basePoint = inCloud.at(i);
		  nsIt = notSeg.begin();

		  while (true) {
			 if (nsIt != notSeg.end()) {
				vector <float> checkPoint = *nsIt;
				if (checkPoint.at(0) - basePoint.at(0) > PT_DIST) break;
				float distBet = distance(basePoint, checkPoint);
				if (distBet < PT_DIST) {
				    inCloud.push_back(checkPoint);
				    nsIt = notSeg.erase(nsIt);
				}
				else nsIt++;

			 }
			 else break;
		  }
	   }
	   objList.push_back(inCloud);
    }
    
    
    return objList;

}



vector<vector<vector<float> > > segmentCloud(vector<vector<float> > toSegment) {
    vector<vector<float> > notSeg=toSegment;
    vector<vector<vector<float> > > objList;

    //generate objects
    while (!notSeg.empty()) {
	   //Setup temp lists (per object)
	   vector<vector<float> > toAdd;
	   vector<vector<float> > inCloud;

	   //Generate one object
	   sortDepth(notSeg);
	   toAdd.push_back(notSeg.at(0));
	   notSeg.erase(notSeg.begin());//O(n)
	   do {
		  //vector<vector<float> > rmToAdd;
		  //vector<vector<float> > rmNotSeg;
		  vector<vector<float> > nextToAdd;
		  BOOST_FOREACH(vector<float> basePoint, toAdd) {
			 BOOST_FOREACH(vector<float> checkPoint, notSeg) {
				//if (basePoint == checkPoint) continue;
				float distBet = distance(basePoint, checkPoint);
				if (distBet < PT_DIST) {
				    nextToAdd.push_back(checkPoint);
				    
				}
				if (checkPoint.at(0) - basePoint.at(0) > PT_DIST) break;
			 }
			 
			 //remove elements in nextToAdd from notSeg
			 BOOST_FOREACH(vector<float> removePoint, nextToAdd) {
				notSeg.erase(std::remove(notSeg.begin(), notSeg.end(), removePoint), notSeg.end());
			 }
		  }

		  //move toAdd into incloud
		  while (!toAdd.empty()) {
			 inCloud.push_back(toAdd.at(0));
			 toAdd.erase(toAdd.begin());
		  }

		  //setup toAdd for next iteration
		  toAdd = nextToAdd;
		   

	   } while (!toAdd.empty());
	   
	   objList.push_back(inCloud);
    }
    return objList;
}

extern "C" {

    int answer() {
	   foo.push_back(3);
	   return foo.at(0);
    }


    int getQuadCenter(float pointCloud[][4], int numNonZero) {
	   if (numNonZero == 0) return -1;

	   vector<vector<float> > notSeg;
	   vector<vector<float> > toAdd;
	   vector<vector<float> > inCloud;

	   /*for (int i = 0; i < 4; i++) {
		  print notSeg[];
	   }*/

	   //transform c array to cpp array
	   for (int i = 0; i < numNonZero; i++) {
		  vector<float> point;
		  for (int j = 0; j < 4; j++) {
			 point.push_back(pointCloud[i][j]);
		  }
		  notSeg.push_back(point);
	   }

	 
	  /* if (DEBUG_PRINT) {
		  sortDepth(notSeg);
		  for (int i = 0; i < 4; i++) {
			 printf("%f ", notSeg.at(0).at(i));
		  }
	   }*/

	   float totalDist = 0;
	   //generate objects
	   while (!notSeg.empty()) {
		  sortDepth(notSeg);
		  toAdd.push_back(notSeg.at(0));
		  notSeg.erase(notSeg.begin());
		  do {
			 BOOST_FOREACH(vector<float> basePoint, toAdd) {
				BOOST_FOREACH(vector<float> checkPoint, notSeg) {
				    if (basePoint == checkPoint) continue;
				    float distBet = distance(basePoint, checkPoint);
				    if (distBet < PT_DIST) {
					   toAdd.push_back(checkPoint);
					   notSeg.erase(std::remove(notSeg.begin(), notSeg.end(), checkPoint), notSeg.end());
				    }
				}		
				inCloud.push_back(basePoint);
				toAdd.erase(std::remove(toAdd.begin(), toAdd.end(), basePoint), toAdd.end());
			 }
		  } while (!toAdd.empty());
	   }

    }
}

int main(int argc, char **argv) {
    //vector<vector<vector<float> > > testCloud = { { { 1,2,3,4 },{ 5,6,7,8 } },{ { 1,1,1,1 },{ 2,2,2,2 } } };
    //writeCloudList(testCloud);
    
    //vector<vector<float> > testCloud = readCloud("cloud.txt");
    //vector<vector<float> > testCloud;
    //for (int i = 0; i < 600*2; i++) {
	   //vector<float> ta = { (float) i,(float) i,(float) i, (float) i };
	   //testCloud.push_back(ta);
    //}

    //for (int i = 610*2; i < 1000*2; i++) {
	   //vector<float> ta = { (float)i,(float)i,(float)i, (float)i };
	   //testCloud.push_back(ta);
    //}

    list<vector<float> > testCloud;
    for (int i = 0; i < 600 * 100; i++) {
	   vector<float> ta = { (float)i,(float)i,(float)i, (float)i };
	   testCloud.push_back(ta);
    }

    for (int i = 610 * 100; i < 1000 * 100; i++) {
	   vector<float> ta = { (float)i,(float)i,(float)i, (float)i };
	   testCloud.push_back(ta);
    }

    //list<vector<float> >  testCloud = {  { 99,99,99,99 },{ 999,999,999,999 }, { 5,5,5,5 }, { 6,6,6,6 },{ 7,7,7,7 } } ;

    writeCloudList(segmentCloudEfficient(testCloud));

    //float* cloudArray = testCloud.data();
    //writeCloud(testCloud);
}
