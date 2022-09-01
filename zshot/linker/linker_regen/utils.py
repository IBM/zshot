def create_input(sentence, max_length, start_delimiter, end_delimiter):
    sent_list = sentence.split(" ")
    if len(sent_list) < max_length:
        return sentence
    else:
        end_delimiter_index = sent_list.index(end_delimiter)
        start_delimiter_index = sent_list.index(start_delimiter)
        half_context = (max_length - (end_delimiter_index - start_delimiter_index)) // 2
        left_index = max(0, start_delimiter_index - half_context)
        right_index = min(len(sent_list), end_delimiter_index + half_context + (
            half_context - (start_delimiter_index - left_index)))
        left_index = left_index - max(0, (half_context - (right_index - end_delimiter_index)))
        print(len(sent_list[left_index:right_index]))
        return " ".join(sent_list[left_index:right_index])
