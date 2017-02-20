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
static const float PT_DIST = 2.0*2.0; //Maximum allowed distance between points of object. Remember to square

vector<int> foo;
vector<vector<vector<float> > > segObjects;

struct Point4 {
    public:
	   float depth, x, y, z;


};


struct depthCmp
{
    bool operator()(const Point4* lhs,
	   const Point4* rhs) const
    {
	   return lhs->depth < rhs->depth;
    }
};


void sortDepth(vector<Point4* > toSort) {
    //sort notSeg by depth
    std::sort(toSort.begin(), toSort.end(), depthCmp());
}

//Calculates distance between 2 points
float distance(Point4& v1, Point4& v2) {
    float xDif = (v1.x - v2.x);
    float yDif=(v1.y - v2.y);
    float zDif=(v1.z - v2.z);
    return xDif*xDif + yDif*yDif + zDif * zDif;
}

void writeCloud(vector<Point4* > cloud, string name) {
    std::ofstream outputFile(name);
    std::ostream_iterator<float> output_iterator(outputFile, " ");
    BOOST_FOREACH(Point4* point, cloud){
	   std::copy(&point->depth, &point->z, output_iterator);
	   outputFile << "\n";
    }
    outputFile.close();
}

void writeCloudList(vector<vector<Point4* > > cloudList ){
    int cloudNum = 0;
    BOOST_FOREACH(vector<Point4* > toWrite, cloudList) {
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

vector<Point4*> generateObject(list<Point4*>& pointList) {
    list<Point4* >::iterator nsIt;
    vector<Point4* >inCloud;
    Point4* closestPoint = pointList.front();
    inCloud.push_back(closestPoint);
    pointList.pop_front();

    for (int i = 0; i < (int)inCloud.size(); i++) {//not at end of incloud

	   Point4& basePoint = *inCloud.at(i);
	   nsIt = pointList.begin();

	   while (true) {
		  if (nsIt != pointList.end()) {
			 Point4& checkPoint = **nsIt;
			 if (checkPoint.depth - basePoint.depth > PT_DIST) break;
			 float distBet = distance(basePoint, checkPoint);
			 if (distBet < PT_DIST) {
				inCloud.push_back(&checkPoint);
				nsIt = pointList.erase(nsIt);
			 }
			 else nsIt++;

		  }
		  else break;
	   }
    }

    return inCloud;
}


vector<vector<Point4* > >  segmentCloudEfficient(float toSegment[][4], int size) {

    list<Point4* > notSeg;
    for (int i = 0; i < size; i++) {
	   notSeg.push_back((Point4*) &(toSegment[i][0]));
    }
    

    vector<vector<Point4* > > objList;

    notSeg.sort(depthCmp());
    
    while (!notSeg.empty()) {	   
	   objList.push_back(generateObject(notSeg));
    }
    
    
    return objList;

}


extern "C" {

    int answer() {
	   foo.push_back(3);
	   return foo.at(0);
    }


    
}

const int numElements = 100000;
float testCloud[numElements][4];

int main(int argc, char **argv) {
    
    for (int i = 0; i < numElements/2; i++) {
	  /* vector<float> ta = { (float)i,(float)i,(float)i, (float)i };
	   testCloud.push_back(ta);*/
	   for (int j = 0; j < 4; j++) {
		  testCloud[i][j] = (float)i;
	   }
    }

    for (int i = (numElements / 2) + 10; i < numElements; i++) {
	   /*vector<float> ta = { (float)i,(float)i,(float)i, (float)i };
	   testCloud.push_back(ta);*/
	   for (int j = 0; j < 4; j++) {
		  testCloud[i][j] = (float)i;
	   }
    }

    //float  testcloud[5][4] = {  { 99,99,99,99 },{ 999,999,999,999 }, { 5,5,5,5 }, { 6,6,6,6 },{ 7,7,7,7 } } ;

    segmentCloudEfficient(testCloud, numElements);

  //  float testA[4] = { 1,2,3,4 };


   
}


float*** convertToCVector(vector<vector<vector<float> > >&vals, int X, int Y, int Z)
{
    float*** temp;
    temp = new float**[X];
    for (int i = 0; (i < X); i++)
    {
	   temp[i] = new float*[Y];
	   for (int j = 0; (j < Y); j++)
	   {
		  temp[i][j] = new float[Z];
		  for (int k = 0; (k < Z); k++)
		  {

			 temp[i][j][k] = vals[i][j][k];
		  }
	   }
    }
    return temp;
}

//int getQuadCenter(float pointCloud[][4], int numNonZero) {
//    if (numNonZero == 0) return -1;
//
//    vector<vector<float> > notSeg;
//    vector<vector<float> > toAdd;
//    vector<vector<float> > inCloud;
//
//    /*for (int i = 0; i < 4; i++) {
//    print notSeg[];
//    }*/
//
//    //transform c array to cpp array
//    for (int i = 0; i < numNonZero; i++) {
//	   vector<float> point;
//	   for (int j = 0; j < 4; j++) {
//		  point.push_back(pointCloud[i][j]);
//	   }
//	   notSeg.push_back(point);
//    }
//
//
//    /* if (DEBUG_PRINT) {
//    sortDepth(notSeg);
//    for (int i = 0; i < 4; i++) {
//    printf("%f ", notSeg.at(0).at(i));
//    }
//    }*/
//
//    float totalDist = 0;
//    //generate objects
//    while (!notSeg.empty()) {
//	   sortDepth(notSeg);
//	   toAdd.push_back(notSeg.at(0));
//	   notSeg.erase(notSeg.begin());
//	   do {
//		  BOOST_FOREACH(vector<float> basePoint, toAdd) {
//			 BOOST_FOREACH(vector<float> checkPoint, notSeg) {
//				if (basePoint == checkPoint) continue;
//				float distBet = distance(basePoint, checkPoint);
//				if (distBet < PT_DIST) {
//				    toAdd.push_back(checkPoint);
//				    notSeg.erase(std::remove(notSeg.begin(), notSeg.end(), checkPoint), notSeg.end());
//				}
//			 }
//			 inCloud.push_back(basePoint);
//			 toAdd.erase(std::remove(toAdd.begin(), toAdd.end(), basePoint), toAdd.end());
//		  }
//	   } while (!toAdd.empty());
//    }
//
//}

//vector<vector<vector<float> > > segmentCloud(vector<vector<float> > toSegment) {
//    vector<vector<float> > notSeg=toSegment;
//    vector<vector<vector<float> > > objList;
//
//    //generate objects
//    while (!notSeg.empty()) {
//	   //Setup temp lists (per object)
//	   vector<vector<float> > toAdd;
//	   vector<vector<float> > inCloud;
//
//	   //Generate one object
//	   sortDepth(notSeg);
//	   toAdd.push_back(notSeg.at(0));
//	   notSeg.erase(notSeg.begin());//O(n)
//	   do {
//		  //vector<vector<float> > rmToAdd;
//		  //vector<vector<float> > rmNotSeg;
//		  vector<vector<float> > nextToAdd;
//		  BOOST_FOREACH(vector<float> basePoint, toAdd) {
//			 BOOST_FOREACH(vector<float> checkPoint, notSeg) {
//				//if (basePoint == checkPoint) continue;
//				float distBet =  distance(basePoint, checkPoint);
//				if (distBet < PT_DIST) {
//				    nextToAdd.push_back(checkPoint);
//				    
//				}
//				if (checkPoint.at(0) - basePoint.at(0) > PT_DIST) break;
//			 }
//			 
//			 //remove elements in nextToAdd from notSeg
//			 BOOST_FOREACH(vector<float> removePoint, nextToAdd) {
//				notSeg.erase(std::remove(notSeg.begin(), notSeg.end(), removePoint), notSeg.end());
//			 }
//		  }
//
//		  //move toAdd into incloud
//		  while (!toAdd.empty()) {
//			 inCloud.push_back(toAdd.at(0));
//			 toAdd.erase(toAdd.begin());
//		  }
//
//		  //setup toAdd for next iteration
//		  toAdd = nextToAdd;
//		   
//
//	   } while (!toAdd.empty());
//	   
//	   objList.push_back(inCloud);
//    }
//    return objList;
//}
