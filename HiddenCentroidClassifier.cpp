#include<iostream>
#include<armadillo>

/// Generate Cluster Data
//  Generate Clusters for training
arma::Mat<double> genCluster(const arma::uword n_points, const arma::uword n_dim, const arma::Col<double> offset, const int classifier)
{
/*
n_points            = Number of points
n_dim               = The number of dimenions of each point
offset              = Offset of the cluster, must by the same dimension as the centroid
classifier          = Symbol for the classifier
*/

    arma::Mat<double> clustData(n_dim, n_points, arma::fill::randn);
    arma::Row<double> classifierType(n_points);
    classifierType.fill(classifier);

    for(int i = 0; i < n_dim; i++)
    {
        clustData.row(i) = clustData.row(i) + offset(i);
    }
    clustData = arma::join_vert(clustData,classifierType);

    return clustData;
}

/// Generate Test Cluster Data
//  Generate Clusters for testing
arma::Mat<double> genTestSet(const arma::uword n_points, const arma::uword n_dim, const arma::Col<double> offset_1, const int classifier_1, arma::Col<double> offset_2, const int classifier_2)
{
/*
n_points            = Number of points
n_dim               = The number of dimenions of each point
offset_1            = Offset of the cluster, must by the same dimension as the centroid
classifier_1        = Symbol for the classifier
offset_2            = Offset of the cluster, must by the same dimension as the centroid
classifier_2        = Symbol for the classifier
*/


    arma::Mat<double> testData(n_dim + 1, n_points, arma::fill::randn);
    arma::Col<double> rndVec(n_points, arma::fill::randu);

    for(int i = 0; i < n_points; i++)
    {
        if(rndVec(i) < 0.5)
        {
            testData(arma::span(0,n_dim-1),i) = testData(arma::span(0,n_dim-1),i) + offset_1;
            testData(n_dim,i) = classifier_1;
        }
        else
        {
            testData(arma::span(0,n_dim-1),i) = testData(arma::span(0,n_dim-1),i) + offset_2;
            testData(n_dim,i) = classifier_2;
        }
    }

    return testData;

}

/// Linear Kernal
//  Computes the dot product of two vectors
double linearKernal(arma::Col<double> v1, arma::Col<double> v2)
{
/*
v1, v2      = Two vectors used for the kernal input
*/
    //Computer dot product
    return arma::dot(v1,v2);
}

///Distance centroid
double distFunction(const arma::Mat<double>& D_points, const arma::Col<double>& T_point)
{
/*
D_points      = Cluster points
T_point       = Testing point
*/
    double term_one;
    double term_two;
    double term_three;

    //Term one in hidden centroid equation
    term_one = linearKernal(T_point, T_point);

    //Term two in hidden centroid equation
    for(int i = 0; i < D_points.n_cols; i++)
    {
        term_two = term_two + linearKernal(D_points.col(i),T_point);
    }
    term_two = -2*(term_two/D_points.n_cols);

    //Term three in hidden centroid equation
    for(int i = 0; i < D_points.n_cols; i++)
    {
        for(int j = 0; j < D_points.n_cols; j++)
        {
            term_three = term_three + linearKernal(D_points.col(i), D_points.col(j));
        }
    }
    term_three = (term_three)*(1/(D_points.n_cols*D_points.n_cols));

    return term_one + term_two + term_three;

}

///Classification function
arma::Col<double> classifyData(const arma::Mat<double>& D_POS, const int D_POS_classifier, const arma::Mat<double>& D_NEG, const int D_NEG_classifier, const arma::Mat<double>& D_TEST)
{
/*
D_POS                = Positive training data
D_POS_classifier     = Positive classifier
D_NEG                = Negative training data
D_NEG_classifier     = Negative classifier
D_TEST               = Test Data
*/

    arma::Col<double> classifiedData(D_TEST.n_cols);

    for(int i =0 ; i < D_TEST.n_cols; i++)
    {
        if(distFunction(D_POS, D_TEST.col(i)) > distFunction(D_NEG, D_TEST.col(i)))
        {
            classifiedData(i) = D_NEG_classifier;
        }
        else
        {
            classifiedData(i) = D_POS_classifier;
        }
    }

    return classifiedData;

}

int main()
{
    /// Settings
    arma::arma_rng::set_seed_random();

    /// Constants
    const arma::uword n_dim {2}; //number of dimensions
    const arma::uword D_POS_points {1000}; // Number of points in the positive data set
    const arma::uword D_NEG_points {1000}; // Number of points in the negative data set
    const arma::uword D_TEST_points {100}; // // Number of points in the test data set
    const int D_POS_classifier {1}; // Classifier symbol for the positive data set
    const int D_NEG_classifier {-1}; // Classifier symbol for the negative data set

    ///Variables
    arma::Col<double> D_POS_offset {0, 0}; // Offset for the positive data set
    arma::Col<double> D_NEG_offset {0,-0}; // Offset for the negative data set
    /*
    arma::Mat<double> D_POS = genCluster(D_POS_points, n_dim, D_POS_offset, D_POS_classifier); // Generate positive data set
    arma::Mat<double> D_NEG = genCluster(D_NEG_points, n_dim, D_NEG_offset, D_NEG_classifier); // Generate negative data set
    arma::Mat<double> D_TEST = genTestSet(D_TEST_points, n_dim, D_POS_offset, D_POS_classifier, D_NEG_offset, D_NEG_classifier); // Generate test data set
    arma::Col<double> D_CLAS; // Classified data
    */
    arma::Mat<double> D_POS; // Generate positive data set
    arma::Mat<double> D_NEG; // Generate negative data set
    arma::Mat<double> D_TEST; // Generate test data set
    arma::Col<double> D_CLAS; // Classified data

    ///Classify test set
   // D_CLAS = classifyData(D_POS.rows(0,n_dim-1),D_POS_classifier, D_NEG.rows(0,n_dim-1), D_NEG_classifier,D_TEST.rows(0,n_dim-1));


    /// Calculate error
    arma::Col<double> D_ERR(10, arma::fill::zeros);

    for(int i = 0; i < 10; i++)
    {
        D_POS_offset = {0.1*i, 0.1*i};
        D_NEG_offset = {-0.1*i, -0.1*i};

        D_POS = genCluster(D_POS_points, n_dim, D_POS_offset, D_POS_classifier); // Generate positive data set
        D_NEG = genCluster(D_NEG_points, n_dim, D_NEG_offset, D_NEG_classifier); // Generate negative data set
        D_TEST = genTestSet(D_TEST_points, n_dim, D_POS_offset, D_POS_classifier, D_NEG_offset, D_NEG_classifier); // Generate test data set


        D_CLAS = classifyData(D_POS.rows(0,n_dim-1),D_POS_classifier, D_NEG.rows(0,n_dim-1), D_NEG_classifier,D_TEST.rows(0,n_dim-1));

       for(int j = 0; j < D_CLAS.n_rows; j++)
       {
            if(D_TEST(n_dim,j) == D_CLAS(j))
            {
                D_ERR(i) = D_ERR(i) + 1;
            }
       }
    }
    D_ERR = D_ERR/10;


    D_ERR.save("ClusterData/D_ERR", arma::csv_ascii);
    D_NEG.save("ClusterData/D_NEG", arma::csv_ascii);
    D_POS.save("ClusterData/D_POS", arma::csv_ascii);
    D_TEST.save("ClusterData/D_TEST", arma::csv_ascii);
    D_CLAS.save("ClusterData/D_CLAS", arma::csv_ascii);
}

