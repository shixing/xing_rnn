import numpy as np
import nltk

def corpus_level_bleu(references, targets):
    # references: [['1','2'],...]
    references = [[x] for x in references]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, targets)
    return bleu_score

def sentence_level_bleu(references, samples):
    # Inputs:
    # reference:
    #   [[1,3,5,2] * n], shape: [n,l1]
    # samples:
    #   shape: [n * r, l2, 1]
    #
    # Outputs:
    # target_inputs, target_outputs, target_weights, bleus

    n = len(references)
    samples = samples.reshape(samples.shape[:-1])
    samples = samples.astype(np.int32)
    r = int(samples.shape[0] / n)

    smooth = nltk.translate.bleu_score.SmoothingFunction()

    target_inputs_ids, target_outputs_ids, target_weights, bleus = [], [], [], []
    
    temp_target_inputs, temp_target_outputs = [],[]

    target_length = 0
    for i in xrange(n):
        reference = references[i]
        ref_str = [str(x) for x in reference[:-1]]
        for j in xrange(r):
            sample = samples[i * r + j]
            if j == 0:
                sample = reference
            sample_str = []
            new_sample = []
            for x in sample:
                new_sample.append(x)
                if x == 2:
                    break
                sample_str.append(str(x))
            
            bleu = nltk.translate.bleu_score.sentence_bleu([ref_str],sample_str, smoothing_function = smooth.method0)
            #print(ref_str)
            #print(sample_str)
            #print(bleu)
            if target_length < len(new_sample):
                target_length = len(new_sample)
            bleus.append(bleu)
            temp_target_inputs.append([1]+new_sample[:-1])
            temp_target_outputs.append(new_sample)

    # pad and add the corresponding weights;
    for i in xrange(len(temp_target_inputs)):
        target_input = temp_target_inputs[i]
        target_output = temp_target_outputs[i]
        l = len(target_output)
        weights = [1.0] * l + [0.0] * (target_length - l)
        target_input_pad = target_input + [0] * (target_length - l)
        target_output_pad = target_output + [0] * (target_length - l)
        target_inputs_ids.append(target_input_pad)
        target_outputs_ids.append(target_output_pad)
        target_weights.append(weights)

    return target_inputs_ids, target_outputs_ids, target_weights, bleus

            
    
if __name__ == "__main__":
    samples = np.array([[[4.0], [14.0], [28.0], [24.0], [23.0], [8.0], [25.0], [27.0], [7.0], [7.0], [24.0], [16.0], [13.0], [2.0], [2.0], [2.0], [2.0], [21.0], [15.0], [2.0], [2.0], [2.0], [2.0], [2.0], [2.0], [29.0], [13.0], [11.0], [2.0], [2.0], [2.0]], [[14.0], [14.0], [25.0], [4.0], [11.0], [16.0], [24.0], [6.0], [7.0], [23.0], [20.0], [13.0], [24.0], [14.0], [2.0], [2.0], [2.0], [2.0], [8.0], [11.0], [2.0], [22.0], [2.0], [15.0], [2.0], [27.0], [2.0], [2.0], [2.0], [2.0], [13.0]], [[14.0], [4.0], [28.0], [8.0], [24.0], [24.0], [7.0], [5.0], [5.0], [15.0], [16.0], [6.0], [21.0], [2.0], [2.0], [2.0], [10.0], [2.0], [2.0], [2.0], [22.0], [28.0], [13.0], [2.0], [2.0], [2.0], [2.0], [15.0], [27.0], [2.0], [2.0]], [[7.0], [14.0], [16.0], [5.0], [13.0], [21.0], [21.0], [18.0], [4.0], [10.0], [24.0], [5.0], [4.0], [20.0], [20.0], [14.0], [12.0], [4.0], [21.0], [26.0], [2.0], [2.0], [2.0], [2.0], [2.0], [2.0], [2.0], [29.0], [2.0], [2.0], [21.0]], [[7.0], [13.0], [5.0], [23.0], [28.0], [21.0], [14.0], [24.0], [29.0], [21.0], [4.0], [9.0], [12.0], [18.0], [10.0], [25.0], [4.0], [6.0], [9.0], [14.0], [2.0], [2.0], [2.0], [2.0], [13.0], [4.0], [2.0], [2.0], [2.0], [2.0], [2.0]], [[7.0], [17.0], [14.0], [4.0], [23.0], [4.0], [5.0], [21.0], [14.0], [8.0], [24.0], [13.0], [10.0], [12.0], [20.0], [4.0], [28.0], [29.0], [6.0], [2.0], [2.0], [2.0], [2.0], [2.0], [9.0], [11.0], [13.0], [2.0], [2.0], [2.0], [2.0]]]).astype(np.int32)
    references = [[14, 4, 28, 25, 24, 24, 15, 5, 14, 24, 25, 4, 23, 2], [7, 16, 27, 13, 18, 21, 23, 24, 21, 21, 4, 4, 20, 8, 5, 29, 20, 5, 4, 9, 2]]
    target_inputs_ids, target_outputs_ids, target_weights, bleus = sentence_level_bleu(references,samples)
    print(samples.reshape(samples.shape[:-1]))
    print(references)
    print(target_inputs_ids)
    print(target_outputs_ids)
    print(target_weights)
    print(bleus)


