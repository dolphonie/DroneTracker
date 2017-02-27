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
static const float MAX_DIST = .15*.15; //Maximum allowed distance between points of object. Remember to square
static const float MIN_DEPTH = .01;//Throw away point if closer
static const float MAX_COORD_VALUE = 30;

//Quad identification constants
static const int MIN_QUAD_PTS = 500;//Minimum number of points for object to be considered a quad
static const float MAX_QUAD_DEPTH = .5;//Maximum depth of quad in meters
static const float MIN_QUAD_WIDTH = .1;
static const float MAX_QUAD_WIDTH = .6;
static const float MIN_QUAD_HEIGHT = .15;
static const float MAX_QUAD_HEIGHT = .6;
static const float NOT_FOUND_SENTINEL_VALUE = -99; //Return if quad not found

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
    //if (DEBUG_PRINT) printf("writeCloud running. 1st depth: %f\n", cloud.at(0)->depth);
    std::ofstream outputFile(name);
    std::ostream_iterator<float> output_iterator(outputFile, " ");
    BOOST_FOREACH(Point4* point, cloud){
	   std::copy(&point->depth, (&point->z)+1, output_iterator);
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

class Float4 {
public:
    float data[4];
};

void readCloud(string fileName, vector<Float4  >& cloud) {
    
    int rowNum = 0;
    std::ifstream inputFile(fileName);
    while (!inputFile.eof()) {
	   
	   Float4 tmpPt;
	   float tmpFloat;

		  for (int j = 0; j < 4; j++) {
			 inputFile >> tmpFloat;
			 //if (DEBUG_PRINT) printf("data: %f\n", tmpFloat);
			 //if (inputFile.eof()) return;
			 tmpPt.data[j] = tmpFloat;
		  }
		  cloud.push_back(tmpPt);
		 // if (DEBUG_PRINT) printf("Row Number: %d\n", ++rowNum);
    }

}

void generateObject(list<Point4*>& pointList, vector<Point4* >& inCloud) {
    list<Point4* >::iterator nsIt;
    
    Point4* closestPoint = pointList.front();
    inCloud.push_back(closestPoint);
    pointList.pop_front();

    for (int i = 0; i < (int)inCloud.size(); i++) {//not at end of incloud

	   Point4& basePoint = *inCloud.at(i);
	   nsIt = pointList.begin();

	   while (true) {
		  if (nsIt != pointList.end()) {
			 Point4& checkPoint = **nsIt;
			 if (checkPoint.depth - basePoint.depth > MAX_DIST) break;
			 float distBet = distance(basePoint, checkPoint);
			 if (distBet < MAX_DIST) {
				inCloud.push_back(&checkPoint);
				nsIt = pointList.erase(nsIt);
			 }
			 else nsIt++;

		  }
		  else break;
	   }
    }

}


void  segmentCloudEfficient(float toSegment[][4], int size, vector<vector<Point4* > >& objList) {
    
    list<Point4* > notSeg;
    for (int i = 0; i < size; i++) {
	   if (toSegment[i][0] < MIN_DEPTH || std::isnan(toSegment[i][0])) continue;	
	   if (toSegment[i][1] > MAX_COORD_VALUE || std::isnan(toSegment[i][1])) continue;
	   if (toSegment[i][2] > MAX_COORD_VALUE || std::isnan(toSegment[i][2])) continue;
	   if (toSegment[i][3] > MAX_COORD_VALUE || std::isnan(toSegment[i][3])) continue;

	   notSeg.push_back((Point4*) &(toSegment[i][0]));
    }
    


    notSeg.sort(depthCmp());
    
    if (DEBUG_PRINT) printf("Generating objects\n");
    while (!notSeg.empty()) {
	   vector<Point4* > solid;
	   generateObject(notSeg, solid);
	   objList.push_back(solid);
    }
    if (DEBUG_PRINT) printf("Objects generated\n");
}

//returns box width and center of dim
void getObjectDims(vector<Point4* >& solid, int dim, float* objDims) {
    float max= -100;//Explore min_value if necessary
    float min = 100;
    float ptSums = 0;
    for (int i = 0; i < solid.size(); i++) {
	   float pos;
	   switch (dim) {
		  case 0: 
			 pos = solid.at(i)->x;
			 break;
		  case 1:
			 pos = solid.at(i)->y;
			 break;
		  case 2:
			 pos = solid.at(i)->z;
			 break;
		  default:
			 break;
	   }
	   ptSums += pos;
	   if (pos > max) max = pos;
	   if (pos < min) min = pos;
    }
    //if(DEBUG_PRINT) printf("sum of points: %f\n" ,ptSums);
    ptSums /= solid.size();
    //if (DEBUG_PRINT) printf("center: %f\n", ptSums);
    //if (DEBUG_PRINT) printf("max: %f\n", max);
    //if (DEBUG_PRINT) printf("min: %f\n", min);
    objDims[0] = max - min;
    objDims[1] = ptSums;
}

//Returns center of quad (for now list of quad objects)
void locateQuad(float* xPtr, float* yPtr, float* zPtr, vector<vector<Point4* > >& objList) {
    //vector<int> potentialQuads;
    for (int i = 0; i < objList.size(); i++) {
	   vector<Point4* >& toCheck = objList.at(i);
	   if (toCheck.size() < MIN_QUAD_PTS) continue;
	   if (toCheck.at(toCheck.size() - 1)->depth - toCheck.at(0)->depth > MAX_QUAD_DEPTH) continue;
	   
	   //get object width
	   float width[2];
	   getObjectDims(toCheck, 0, width);
	   if (DEBUG_PRINT) printf("width: %f\n", width[0]);
	   if (width[0]<MIN_QUAD_WIDTH || width[0]>MAX_QUAD_WIDTH) continue;


	   float height[2];
	   getObjectDims(toCheck, 1, height);
	   if (DEBUG_PRINT) printf("height: %f\n", height[0]);
	   if (height[0]<MIN_QUAD_HEIGHT || height[0]>MAX_QUAD_HEIGHT) continue;

	   //if(DEBUG_PRINT) printf("quad loc: %f, %f", width[1], height[1]);

	   *zPtr = toCheck.at(i)->z;
	   *xPtr = width[1];
	   *yPtr = height[1];
	   return;
	   //potentialQuads.push_back(i);

    }

    *xPtr = NOT_FOUND_SENTINEL_VALUE;
    *yPtr = NOT_FOUND_SENTINEL_VALUE;
    *zPtr = NOT_FOUND_SENTINEL_VALUE;

    //return potentialQuads;
}

extern "C" {

    int answer() {
	   foo.push_back(3);
	   return foo.at(0);
    }

    void segmentQuad(float toSegment[][4], int size, float*xLoc, float* yLoc, float* zLoc ) {
	   vector<vector<Point4* > > segCloud;
	   if (DEBUG_PRINT) printf("cpp called\n");
	   segmentCloudEfficient(toSegment, size, segCloud);
	   if (DEBUG_PRINT) printf("cloud segmented\n");
	   //if(DEBUG_PRINT) writeCloudList(segCloud);
	   locateQuad(xLoc, yLoc, zLoc, segCloud);
	   if (DEBUG_PRINT) printf("quad located\n");
    }
    
}

const int numElements = 100000;
float testCloud[numElements][4];

int main(int argc, char **argv) {

   // for (int i = 0; i < numElements/2; i++) {
	  ///* vector<float> ta = { (float)i,(float)i,(float)i, (float)i };
	  // testCloud.push_back(ta);*/
	  // for (int j = 0; j < 4; j++) {
		 // testCloud[i][j] = (float)i;
	  // }
   // }

   // for (int i = (numElements / 2) + 10; i < numElements; i++) {
	  // /*vector<float> ta = { (float)i,(float)i,(float)i, (float)i };
	  // testCloud.push_back(ta);*/
	  // for (int j = 0; j < 4; j++) {
		 // testCloud[i][j] = (float)i;
	  // }
   // }

    //float  testcloud[5][4] = {  { 99,99,99,99 },{ 999,999,999,999 }, { 5,5,5,5 }, { 6,6,6,6 },{ 7,7,7,7 } } ;

    typedef float float4[4];
    
    vector<Float4> ptVectors;
    readCloud("img.txt", ptVectors);
    float4* foo = (float4*) ptVectors.data();

    float xPrint = 0;
    float yPrint = 0;
    float zPrint = 0;
    segmentQuad(foo,ptVectors.size(), &xPrint,&yPrint, &zPrint);
    
    if (DEBUG_PRINT) printf("quad loc: %f, %f", xPrint, yPrint);  
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
