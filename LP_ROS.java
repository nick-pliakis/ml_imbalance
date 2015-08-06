/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ergasia2pkg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * Implementation of the LabelPowerSet Random Oversampling Algorithm described
 * in: Addressing imbalance in multi label classification: Measures and random
 * resampling algorithms
 *
 * @author Panos
 */
public class LP_ROS {

    private float P; //precentage

    public LP_ROS(float P) {
        this.P = P;

    }

    /**
     * A method that implements the Label Powerset Oversampling method to reduce
     * imbalance in Multilabel data.
     *
     * @param MultiLabelInstances the data to be handled
     * @return MultiLabelInstances data with reduced imbalance
     * @throws Exception to be handled on an upper level
     *
     */
    public MultiLabelInstances transformInstances(MultiLabelInstances mlData) throws Exception {

        int samplesToClone = (int) (mlData.getNumInstances() * P);
        HashMap labelsetGroups = groupByLabelSet(mlData);
        //calculate meansize
        double labelsetNum = labelsetGroups.size();
        double meansize = ((1 / labelsetNum) * mlData.getNumInstances());
        Set<String> keyset = (Set<String>) labelsetGroups.keySet();

        HashMap minBag = new HashMap<String, List<Instance>>();
        TreeMap<Integer, List<String>> sortedBagIndex = new TreeMap<Integer, List<String>>(Collections.reverseOrder());
        for (String labelset : keyset) {
            List instanceList = (List) labelsetGroups.get(labelset);
            if (instanceList.size() < meansize) {
                minBag.put(labelset, instanceList);
                if (sortedBagIndex.containsKey(instanceList.size())) {
                    List templist = sortedBagIndex.get(instanceList.size());
                    templist.add(labelset);
                    sortedBagIndex.put(instanceList.size(), templist);
                } else {
                    List templist = new ArrayList<String>();
                    templist.add(labelset);
                    sortedBagIndex.put(instanceList.size(), templist);
                }
            }
        }

        double meanInc = (double) samplesToClone / minBag.size();

        int bagsProccessed = 0;
        int samplesCloned = 0;
        //int distributedRemainder=0;
        int[] rBag = new int[minBag.size()];
        for (Integer bagsize : sortedBagIndex.keySet()) {

            List<String> currentLabelsets = (ArrayList<String>) sortedBagIndex.get(bagsize);
            for (String labelset : currentLabelsets) {
                int instances_to_clone_NUM = (int) Math.max((meansize - bagsize), meanInc) + 1;

                List cleanList;
                int suplement;
                if ((rBag[bagsProccessed] + bagsize) > meansize) {

                    suplement = instances_to_clone_NUM;
                    if ((samplesCloned + rBag[bagsProccessed]) > samplesToClone) {
                        break;
                    }
                    samplesCloned += rBag[bagsProccessed];
                    cleanList = cloneInstances((List) minBag.get(labelset), rBag[bagsProccessed]);

                    if (bagsProccessed != minBag.size() - 1) {
                        distribute(suplement, rBag, bagsProccessed + 1);
                    }
                } else {

                    if ((samplesCloned + rBag[bagsProccessed] + instances_to_clone_NUM) > samplesToClone) {
                        break;
                    }

                    samplesCloned += rBag[bagsProccessed] + instances_to_clone_NUM;
                    cleanList = cloneInstances((List) minBag.get(labelset), instances_to_clone_NUM + rBag[bagsProccessed]);
                }
                bagsProccessed++;
                minBag.put(labelset, cleanList);

            }
        }

        labelsetGroups.putAll(minBag);
        mlData = createNewMultilabelInstance(labelsetGroups, mlData);
        return mlData;

    }

    /**
     * Distributes remaining elements into bags
     *
     * @param int remaining element
     * @param int[] a bag
     * @param int position to distribute elements
     */
    private void distribute(int remainder, int[] rBag, int position) {
        int i = position;

        while (remainder != 0) {

            rBag[i]++;

            remainder--;

            i++;
            if (i == rBag.length) {
                i = position;
            }
        }
    }

    /**
     * Clones a list of instances
     *
     * @param List of instances
     * @param int number of instances to clone
     * @return List<Instances> cloned list of instances
     */
    private List<Instances> cloneInstances(List mylist, int cloneNumber) {

        Random rn = new Random();
        for (int i = 0; i < cloneNumber; i++) {
            int rIndex = rn.nextInt(mylist.size());
            Instance inst = (Instance) mylist.get(rIndex);
            mylist.add(inst);
        }
        return mylist;
    }

    /**
     * Creates a new MultiLabelInstances object given a list of Instance
     *
     * @param Hashmap<String,List<Instance> map from which to create instances
     * @param MultiLabelInstances used just to get the Label metadata
     * @return MultiLabelInstances new MultiLabelInstances Object
     */
    private MultiLabelInstances createNewMultilabelInstance(HashMap<String, List<Instance>> labelsetGroup, MultiLabelInstances mlData) throws InvalidDataFormatException {

        Instances in = mlData.getDataSet();
        Enumeration enumeration = in.enumerateAttributes();
        ArrayList attlist = Collections.list(enumeration);
        int capacity = 0;
        for (String labelset : labelsetGroup.keySet()) {
            capacity += labelsetGroup.get(labelset).size();
        }

        Instances newInstances = new Instances("sampledDataset", attlist, capacity);
        for (String labelset : labelsetGroup.keySet()) {
            List<Instance> instanceList = (ArrayList<Instance>) labelsetGroup.get(labelset);
            for (Instance inst : instanceList) {
                newInstances.add(inst);
            }
        }

        MultiLabelInstances newData = new MultiLabelInstances(newInstances, mlData.getLabelsMetaData());
        return newData;
    }

    /**
     * Groups instances by their labels
     *
     * @param MultilabelInstances labeled instances
     * @return HashMap<String,List<Instance>> returned Hashmap with grouping
     */
    public HashMap groupByLabelSet(MultiLabelInstances mlData) {

        Instances inst = mlData.getDataSet();

        Set<Attribute> atts = mlData.getLabelAttributes();
        HashMap LabelSetGroups = new HashMap<String, List<Instance>>();

        for (int i = 0; i < inst.numInstances(); i++) {
            Instance in = inst.get(i);
            String labelsetName = "";
            for (Attribute att : atts) {
                if (in.value(att) != 0) {
                    labelsetName = labelsetName + att.name();
                }

            }
            if (LabelSetGroups.containsKey(labelsetName)) {
                List myList = (List) LabelSetGroups.get(labelsetName);
                myList.add(in);
                LabelSetGroups.put(labelsetName, myList);
            } else {
                List<Instance> myList = new ArrayList<Instance>();
                myList.add(in);
                LabelSetGroups.put(labelsetName, myList);
            }

        }

        return LabelSetGroups;
    }

}
