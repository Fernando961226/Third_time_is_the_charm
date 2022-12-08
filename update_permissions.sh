find . -type d -exec chmod 770 {} \;
find . -type f -exec chmod 660 {} \;
find . -type f -name '*.sh' -exec chmod 770 {} \;
