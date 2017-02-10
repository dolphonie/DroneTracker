#include <vector>
#include <iostream>
#include <utility>
#include <algorithm> 
#include <stdio.h>
#include <math.h>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

using std::vector;

static const bool DEBUG_PRINT = false;
static const bool PT_DIST = false; //Maximum allowed distance between points of object

vector<int> foo;
vector<vector<vector<float> > > segObjects;

int main(int argc, char **argv) {

}

struct depthCmp
{
    bool operator()(const vector<float>& lhs,
	   const vector<float>& rhs) const
    {
	   return lhs[0] < rhs[0];
    }
};

void sortDepth(vector<float> toSort) {
    //sort notSeg by depth
    std::sort(toSort.begin(), toSort.end(), depthCmp());
}

//Calculates distance between 2 points
float distance(vector<float> v1, vector<float> v2) { 
    float sum = 0;
    for (int i = 1; i < 4; i++) {
	   sum += pow(v1.at(i) - v2.at(i),3.0) ;
    }
    return sqrt(sum);
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
	   vector<vector<float> > seg;

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

	 
	   if (DEBUG_PRINT) {
		  sortDepth();
		  for (int i = 0; i < 4; i++) {
			 printf("%f ", notSeg.at(0).at(i));
		  }
	   }
	   
	   float totalDist = 0;
	   //generate objects
	   while (!notSeg.empty()) {
		  sortDepth();
		  toAdd.push_back(std::move(notSeg.at(0)));
		  do {
			 BOOST_FOREACH(vector<float> point, notSeg) {
				//totalDist += distance();
				break;
			 }
		  } while (!toAdd.empty());
	   }


	   //return notSeg.at(0).at(0);
	   return notSeg.size();
	   
	   /*vector<int> foo;
	   foo.push_back(1);
	   foo.push_back(2);

	   return foo.size();*/
    }
}

