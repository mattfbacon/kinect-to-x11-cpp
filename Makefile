.PHONY: build clean format

TARGET_DIR := target
SRC_DIR := src
INCLUDE_DIR := include

build: $(TARGET_DIR)/Makefile
	cd $(TARGET_DIR) && make

format:
	clang-format -i $(SRC_DIR)/** $(INCLUDE_DIR)/**

clean:
	rm -rf $(TARGET_DIR)

$(TARGET_DIR)/Makefile: CMakeLists.txt | $(TARGET_DIR)
	cd $(TARGET_DIR) && cmake -H. -S.. -B.

$(TARGET_DIR):
	mkdir $(TARGET_DIR)
