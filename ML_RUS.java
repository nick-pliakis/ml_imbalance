/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ergasia2pkg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author nickfortress
 */
public class ML_RUS extends RandomSamplerBase {

    /**
     * Class constructor
     *
     * @param P the percentage of the labels to be undersampled (i.e. how many
     * labels the algorithm should remove from the dataset relative to the total
     * dataset size)
     */
    ML_RUS(Float P) {
        super(P);
    }

    /**
     * Method to perform undersampling on the initial dataset. The method
     * removes instances from the dataset according to the algorithm proposed on
     * the paper, utilising the Mean Imbalance Ratio measure.
     *
     * @param mlData MultiLabelInstances object, holds a set of multilabel
     * instances
     * @return MultiLabelInstances object containing the initial labels minus
     * the labels removed by undersampling
     * @throws Exception
     */
    @Override
    public MultiLabelInstances transformInstances(MultiLabelInstances mlData) throws Exception {
        //Initialise the label counters
        labelCount(mlData);
        //Clone the dataset into a new object
        MultiLabelInstances mlDataClone = mlData.clone();
        //Clone a new set to contain all the instances that will be returned
        Instances mlDataReturned = mlData.clone().getDataSet();
        mlDataReturned.delete();

        //Calculate the number of samples to remove
        int samplesToDelete = (int) (mlData.getNumInstances() / (100 * P));
        int remainingLabels;
        //Declare two lists of lists, a minorityBag and a majorityBag. The minBag 
        //will contain lists (bags) of instances having labels with 
        //an imbalance ratio higher than the mean imbalance ratio. These will be 
        //set aside and not tampered with in any way. The majBag will also contain 
        //lists of instances having labels with an imbalance ratio lower than or 
        //equal to the mean imbalance ratio. These instances will be the candidates 
        //for deletion.
        List<List<Instance>> minBags = new ArrayList<>();
        List<List<Instance>> majBags = new ArrayList<>();
        //Get an array with the indices of all the labels
        int L[] = mlDataClone.getLabelIndices();
        //Calculate the dataset's mean imbalance ratio
        double meanIR = meanImbalanceRatio(mlDataClone);
        String labelName;
        int i = 0, m = 0, x, labelCounter = 0;
        //Declare a boolean array which will follow the labelset L, and determine 
        //whether or not a label's instances should be considered for undersampling
        //Initialise all its values to true.
        boolean included[] = new boolean[L.length];
        for (int k = 0; k < L.length; k++) {
            included[k] = true;
        }
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
                        //Remove the label from the dataset
                        mlDataClone.getDataSet().delete(l);
                    }
                }
                //Set the included flag as false, so that the label is not added
                //to the majbags
                included[labelCounter] = false;
                i++;
            }
            labelCounter++;
        }
        //For every label again
        for (int label : L) {
            //Add a new majbag (one for each label)
            majBags.add(new ArrayList<Instance>());
            //Add all the instances having this label to the majbag. Note that 
            //this operation takes place on the cloned dataset, which now contains
            //only the instances not having minority labels
            for (int l = 0; l < mlDataClone.getNumInstances(); l++) {
                if (mlDataClone.getDataSet().get(l).value(label) == 1.0) {
                    majBags.get(m).add(mlDataClone.getDataSet().get(l));
                }
            }
            m++;
        }
        remainingLabels = L.length - minBags.size();

        //While we haven't deleted all the samples yet and we still have labels 
        //to delete
        while (samplesToDelete > 0 && remainingLabels > 0) {
            //For each of the INITIAL labels (not only the ones in the cloned dataset)
            for (int j = 0; j < mlData.getNumLabels(); j++) {
                if (included[j]) {
                    //if it is to be included (meaning it is a majority label), check 
                    //if this bag contains instances. If it doesn't, decrease the 
                    //numbers and go to the next iteration
                    if (majBags.get(j).size() == 0) {
                        included[j] = false;
                        remainingLabels--;
                        continue;
                    }
                    //Get a random instance from the bag
                    x = rand.nextInt(majBags.get(j).size());
                    //Based on the instance and the index, get its label
                    labelName = majBags.get(j).get(x).attribute(L[j]).name();
                    //Remove the instance from the bag
                    majBags.get(j).remove(x);
                    //If the imbalance ratio of the label has increased beyond the 
                    //acceptable limit of the mean imbalance ratio, remove this 
                    //majbag from future candidates
                    if (imbalanceRatioPerLabel(mlDataClone, labelName) >= meanIR) {
                        included[j] = false;
                        remainingLabels--;
                    }
                    samplesToDelete--;
                }
            }
        }
        //Add the contents of the minbags and the majbags to an empty dataset 
        //and return it
        for (List<Instance> list : minBags) {
            for (Instance inst : list) {
                mlDataReturned.add(inst);
            }
        }
        for (List<Instance> list : majBags) {
            for (Instance inst : list) {
                mlDataReturned.add(inst);
            }
        }

        return new MultiLabelInstances(mlDataReturned, mlData.getLabelsMetaData());
    }
}
