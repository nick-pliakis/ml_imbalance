package ergasia2pkg;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;

/**
 * Abstract class providing needed functionalities to our other classes.
 * Provides an initialisation of the labelCounts hash map, containing the
 * occurences of each label, as well as calculations for the imbalance ratio per
 * label, the mean imbalance ratio of the dataset, and the CVIR measure.
 * Provides an abstract method transformInstances, to be implemented by the
 * child classes.
 *
 * @author nickfortress
 */
abstract class RandomSamplerBase implements Serializable {

    //Percentage of dataset to be over/undersampled
    protected Float P = 0.3f;
    //Label counts hash map, mapping labels to number of occurences
    HashMap<String, Double> labelCounts = new HashMap<>();

    /**
     * Constructor
     *
     * @param P Percentage of dataset to be over/undersampled
     */
    RandomSamplerBase(Float P) {
        this.P = P;
    }

    /**
     * Method to perform under/oversampling on the initial dataset. The method
     * must be implemented on any classes that inherit from this one.
     *
     * @param mlData MultiLabelInstances object, holds a set of multilabel
     * instances
     * @return MultiLabelInstances object containing the initial labels minus
     * the labels removed by undersampling
     * @throws Exception
     */
    public abstract MultiLabelInstances transformInstances(MultiLabelInstances mlData) throws Exception;

    /**
     * Calculate the CVIR measure. Utilises the mean imbalance ratio measure and
     * the imbalance measure per label to calculate the total CVIR for the
     * dataset. The calculation formula has been taken from the paper we
     * studied.
     *
     * @param mlData The dataset on which we need to calculate the CVIR
     * @return Double number representing the CVIR measure for this dataset
     * @throws Exception
     */
    public Double calculateCVIR(MultiLabelInstances mlData) throws Exception {
        Double cvir = 0.0;
        Double meanIR = meanImbalanceRatio(mlData);
        String labelNames[] = mlData.getLabelNames();
        for (String lbl : labelNames) {
            cvir += (Math.pow((imbalanceRatioPerLabel(mlData, lbl) - meanIR), 2)) / (labelNames.length - 1);
        }

        return Math.sqrt(cvir) / meanIR;
    }

    /**
     * Calculate the mean imbalance ratio. The whole dataset provided is
     * considered and the mean imbalance ratio takes every instance and label
     * into account.
     *
     * @param mlData The dataset on which we need to calculate the mean
     * imbalance ratio
     * @return Double number representing the mean imbalance ratio of the
     * dataset
     * @throws Exception
     */
    public Double meanImbalanceRatio(MultiLabelInstances mlData) throws Exception {
        String labelNames[] = mlData.getLabelNames();
        Double meanIR = 0.0;
        for (String lbl : labelNames) {
            meanIR += imbalanceRatioPerLabel(mlData, lbl);
        }
        return meanIR / labelNames.length;
    }

    /**
     * Calculate imbalance ratio per label. Takes into account only one label
     * and calculates the imbalance ratio for that label only.
     *
     * @param mlData The dataset containing the label
     * @param label The label we want to calculate the imbalance ratio for
     * @return Double number representing the imbalance ratio for this label
     * only
     * @throws Exception
     */
    public Double imbalanceRatioPerLabel(MultiLabelInstances mlData, String label) throws Exception {
        Double imbalanceRatio;

        double max = 0.0;
        for (Map.Entry<String, Double> entry : labelCounts.entrySet()) {
            if (entry.getValue() > max) {
                max = entry.getValue();
            }
        }

        imbalanceRatio = max / labelCounts.get(label);

        return imbalanceRatio;
    }

    /**
     * Initialise the hash map labelCounts. This maps each label of the dataset
     * to the number of its occurences. This is needed in our other methods.
     *
     * @param mlData The dataset
     * @throws Exception
     */
    protected void labelCount(MultiLabelInstances mlData) throws Exception {
        Set<Attribute> lblAttrs = mlData.getLabelAttributes();
        Instance inst;

        for (int i = 0; i < mlData.getNumInstances(); i++) {
            for (Attribute attr : lblAttrs) {
                if (labelCounts.containsKey(attr.name())) {
                    labelCounts.replace(attr.name(), labelCounts.get(attr.name()) + mlData.getDataSet().get(i).value(attr));
                } else {
                    labelCounts.put(attr.name(), mlData.getDataSet().get(i).value(attr));
                }
            }
        }
    }
}
