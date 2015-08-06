/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ergasia2pkg;

import java.util.ArrayList;
import java.util.List;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import java.util.Random;

/**
 *
 * @author nickfortress
 */
public class ML_ROS extends RandomSamplerBase {
    /**
     * Class constructor
     * @param P the percentage of the labels to be oversampled (i.e. how many labels
     *          the algorithm should add to the dataset relative to the total dataset 
     *          size)
     */
    ML_ROS(Float P) {
        super(P);
    }
    /**
     * Method to perform oversampling on the initial dataset. The method adds instances to the
     * dataset according to the algorithm proposed on the paper, utilising the Mean Imbalance 
     * Ratio measure.
     * @param mlData        MultiLabelInstances object, holds a set of multilabel instances
     * @return              MultiLabelInstances object containing the initial labels and the
     *                      labels added by oversampling
     * @throws Exception 
     */
    @Override
    public MultiLabelInstances transformInstances(MultiLabelInstances mlData) throws Exception {
        //Initialise the label counters
        labelCount(mlData);
        //Clone the dataset into a new object
        MultiLabelInstances mlDataClone = mlData.clone();
        //Calculate the number of samples to clone
        int samplesToClone = (int) (mlDataClone.getNumInstances()/(100*P));
        
        //Declare the list of lists that will hold the minBags. Minbags will be lists that hold
        //lists of instances
        List<List<Instance>> minBags = new ArrayList<>();
        //Index labels array
        int L[] = mlDataClone.getLabelIndices();
        ArrayList<Integer> labInds = new ArrayList<>();
        //Calculate the meanImbalanceRatio for the entire dataset
        double meanIR = meanImbalanceRatio(mlDataClone);
        String labelName;
        int i = 0, x;
        
        Random rand = new Random();
        //Perform the following operation for each label
        //Note that labels are represented by their integer index, which is then
        //transformed to its string name. This was done to avoid problems and 
        //exceptions thrown by methods required below
        for (int label : L) {
            //Get the label name from the current instance, based on label index
            labelName = mlDataClone.getDataSet().attribute(label).name();
            if (imbalanceRatioPerLabel(mlDataClone, labelName) > meanIR) {
                //if the imbalance ratio of the label is greater than the mean 
                //imbalance ratio of the dataset, add it to the minbag corresponding 
                //to the specific label. 
                minBags.add(new ArrayList<Instance>());
                //Add all instances containing this label to the minbag we just 
                //created
                for (int l = 0; l < mlDataClone.getNumInstances(); l++) {
                    if (mlDataClone.getDataSet().get(l).value(label) == 1.0) {
                        minBags.get(i).add(mlDataClone.getDataSet().get(l));
                    }
                }
                //Add the label to the arraylist following the minbags
                labInds.add(label);
                i++;
            }
        }
        //Convert the ArrayList to a regular array
        Integer labIndsArray[] = new Integer[labInds.size()]; 
        labInds.toArray(labIndsArray);
        
        while (samplesToClone > 0) {
            //While we still need to clone samples, we check every minbag
            for (int j = 0; j < minBags.size(); j++) {
                if (labIndsArray[j] != -1) {
                    //If the minbag label index is not set to -1 (meaning it hasn't
                    //been removed from the process, see below), find a random instance 
                    //from this minbag and add it to the dataset.
                    x = rand.nextInt(minBags.get(j).size());
                    mlDataClone.getDataSet().add(minBags.get(j).get(x));
                    //Get the label name based on its counter
                    labelName = mlDataClone.getDataSet().attribute(labIndsArray[j]).name();
                    if (imbalanceRatioPerLabel(mlDataClone, labelName) <= meanIR) {
                        //If its imbalance ratio has dropped to an acceptable level (below 
                        //the mean imbalance ratio of the dataset) then don't pick samples
                        //from this minbag from now on (shown by setting the label index to 
                        //-1
                        labIndsArray[j] = -1;
                    }
                    //Reduce number of samples to clone by one
                    samplesToClone--;
                }
            }
        }
        //Return the final dataset with the cloned instances
        return mlDataClone;
    }
}
