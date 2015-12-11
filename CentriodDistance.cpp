#include<iostream>
#include<armadillo>

/*
Step1. Generate Clusters
Step2. Classify each cluster
Step3. Take a new point, and assign it to a cluster based on its weight
*/

///Generate Clusters
arma::Mat<double> genCluster(const arma::uword nPoints, const arma::uword nDim, const arma::uword nCluster, const arma::uword separation)
{
/*
nPoints             = Number of points in each cluster
nDim                = The number of dimenions of each point
nCluster            = Number of clusters
separation          = The separation of ecah cluster
*/
    arma::Mat<double> dataPoints;

    for(int i =0; i < nCluster; i++)
    {

        arma::Mat<double> tempPoints(nDim, nPoints, arma::fill::randn);
        arma::Row<double> cluster(nPoints);

        //cluster.fill(i);
        tempPoints += i*(separation/nCluster)/2;
        //tempPoints = arma::join_vert(tempPoints,cluster);
        dataPoints = arma::join_horiz(dataPoints, tempPoints);
    }

    dataPoints.save("ClusterData/clusters", arma::csv_ascii);
    return dataPoints;
}

///Linear Kernal
double linearKernal(arma::Col<double> v1, arma::Col<double> v2)
{
/*
v1, v2      = Two vectors used for the kernal input
*/
    //Computer dot product
    return arma::dot(v1,v2);
}


///Distance centroid
double distCentroid(const arma::Mat<double>& clustData, const arma::Col<double> point)
{
/*
data                = Trainin Data
point               = Point to be classified
*/
    double termTwo {0};
    double termThree {0};

    for(int i = 0; i < clustData.n_cols; i++)
    {
        termTwo = termTwo + linearKernal(point, clustData.col(i));
    }

    termTwo = (-termTwo/clustData.n_cols)*2;

    for(int i = 0 ; i< clustData.n_cols ; i++)
    {
        for(int j = 0; j < clustData.n_cols ; j++)
        {
            termThree =  termThree + linearKernal(clustData.col(i), clustData.col(j));
        }
    }

    termThree = (1./(clustData.n_cols*clustData.n_cols))*termThree;

    return (linearKernal(point,point) + termTwo + termThree);
}

///Classification function
void classifyPoint(const arma::Mat<double>& data, const arma::Col<double> point)
{
/*
data                = Trainin Data
point               = Point to be classified
nCluster            = Number of clusters
*/

    double clust0 = distCentroid(data.cols(0, data.n_cols/2-1), point);
    double clust1 = distCentroid(data.cols(data.n_cols/2, data.n_cols-1), point);
    if( clust0 < clust1)
    {
        std::cout << 0 << std::endl;
    }
    else
    {
        std::cout << 1 << std::endl;
    }
        std::cout << clust0 << ", " << clust1;


}


int main()
{

    arma::Mat<double> clusterData;
    clusterData = genCluster(10, 2, 2, 20);
    arma::Col<double> point = { 2, 2};
    classifyPoint(clusterData, point);
    std::cout << std::endl;
    classifyPoint(clusterData, {0, 0});
    std::cout << std::endl;
    classifyPoint(clusterData, {6, 6});


}
